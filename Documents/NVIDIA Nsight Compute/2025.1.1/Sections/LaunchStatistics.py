# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math

import NvRules
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser

requested_metrics = [
    MetricRequest("launch__block_size", "block_size", Importance.OPTIONAL, None, False),
    MetricRequest("launch__grid_size", "grid_size", Importance.OPTIONAL, None, False),
    MetricRequest("device__attribute_multiprocessor_count"),
    MetricRequest("launch__waves_per_multiprocessor", "num_waves", Importance.OPTIONAL, None, False),
]

WARP_SIZE = 32
HARDWARE_MODEL_REF = (
    "See the @url:Hardware Model:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model@"
    " description for more details on launch configurations."
)


def get_identifier():
    return "LaunchConfiguration"


def get_name():
    return "Launch Configuration"


def get_description():
    return "Kernel launch configuration analysis"


def get_section_identifier():
    return "LaunchStats"


def get_parent_rules_identifiers():
    return ["SOLBottleneck"]


def get_problematic_block_sizes(block_size_metric):
    has_block_size_issue = False
    problematic_block_sizes = []

    if block_size_metric.num_instances() == 0:
        # legacy case, block size is not instanced per launch
        has_block_size_issue = block_size_metric.value() % WARP_SIZE != 0
    else:
        for instance_id in range(block_size_metric.num_instances()):
            block_size = block_size_metric.value(instance_id)
            if block_size % WARP_SIZE != 0:
                has_block_size_issue = True
                launch_id = block_size_metric.correlation_ids().value(instance_id)
                problematic_block_sizes.append((launch_id, block_size))

    return has_block_size_issue, problematic_block_sizes


def get_estimated_speedup_block_size(block_size):
    num_warps = math.ceil(block_size / WARP_SIZE)
    num_threads_last_warp = block_size % WARP_SIZE

    if num_threads_last_warp == 0 or num_warps == 0:
        improvement_percent = 0
    else:
        improvement_percent = (
            (1 / num_warps) * (1 - num_threads_last_warp / WARP_SIZE) * 100
        )

    return NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent


def get_max_estimated_speedup_block_size(block_sizes):
    max_speedup = 0
    launch_id_at_max_speedup = 0
    block_size_at_max_speedup = 0

    for launch_id, block_size in block_sizes:
        _, speedup_value = get_estimated_speedup_block_size(block_size)
        if speedup_value > max_speedup:
            max_speedup = speedup_value
            launch_id_at_max_speedup = launch_id
            block_size_at_max_speedup = block_size

    # TODO: We could get a global estimate by weighing the speedup by the
    # relative duration of this launch
    return NvRules.IFrontend.SpeedupType_LOCAL, max_speedup, launch_id_at_max_speedup, block_size_at_max_speedup


def apply_block_size_rule(fe, action, metrics):
    """Execute the block size rule and generate a rule output if necessary.

    Check whether the block size of (each) kernel launch is a multiple of the warp size,
    and generate a rule message and a speedup estimate otherwise.
    In case of range results, find and report the launch with the largest potential speedup.
    """
    rule_name = "Block Size"
    block_size_metric = metrics["block_size"]

    if block_size_metric is None:
        # Currently, light range results do not collect launch__block_size in all cases 
        # and are not supported.
        return

    has_block_size_issue, problematic_block_sizes = \
        get_problematic_block_sizes(block_size_metric)

    if not has_block_size_issue:
        return

    if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
        block_size = int(block_size_metric.value())
        workload_specific_part = "This kernel launch is configured to execute {:d} threads per block.".format(block_size)
        speedup_type, speedup_value = get_estimated_speedup_block_size(block_size)
    else:
        speedup_type, speedup_value, launch_id, block_size = \
            get_max_estimated_speedup_block_size(problematic_block_sizes)
        workload_specific_part = (
            "Some kernel launches of this workload are configured to use a number"
            " of threads per block that is not a multiple of {}"
            " (e.g., {} {} for launch ID {}).".format(
                WARP_SIZE,
                block_size,
                "thread" if block_size == 1 else "threads",
                launch_id,
            )
        )

    rule_message = (
        "Threads are executed in groups of {} threads called warps. {}"
        " Consequently, some threads in a warp are masked off and those hardware resources are unused."
        " Try changing the number of threads per block to be a multiple of {} threads. Between 128 and 256 threads per block is a good initial range for experimentation."
        " Use smaller thread blocks rather than one large thread block per multiprocessor if latency affects performance. "
        " This is particularly beneficial to kernels that frequently call __syncthreads()."
        " {}".format(
            WARP_SIZE,
            workload_specific_part,
            WARP_SIZE,
            HARDWARE_MODEL_REF
            )
    )

    msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, rule_message, rule_name)
    fe.speedup(msg_id, speedup_type, speedup_value)
    fe.focus_metric(msg_id, block_size_metric.name(), block_size, NvRules.IFrontend.Severity_SEVERITY_LOW, "Arrange the number of threads per block to be a multiple of {}".format(WARP_SIZE))


def get_estimated_speedup_grid_size(grid_size, num_sms):
    # assume the workload scales perfectly with the number of SMs used
    improvement_percent = (num_sms - grid_size) / num_sms * 100

    return NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent


def get_max_estimated_speedup_grid_size(grid_size_metric, num_sms):
    max_speedup = 0
    launch_id_at_max_speedup = 0
    grid_size_at_max_speedup = 0

    for instance_id in range(grid_size_metric.num_instances()):
        grid_size = grid_size_metric.value(instance_id)
        _, speedup_value = get_estimated_speedup_grid_size(grid_size, num_sms)
        if speedup_value > max_speedup:
            max_speedup = speedup_value
            launch_id_at_max_speedup = grid_size_metric.correlation_ids().value(instance_id)
            grid_size_at_max_speedup = grid_size

    # TODO: We could get a global estimate by weighing the speedup by the
    # relative duration of this launch
    return NvRules.IFrontend.SpeedupType_LOCAL, max_speedup, launch_id_at_max_speedup, grid_size_at_max_speedup


def apply_grid_size_rule(fe, action, metrics):
    """Execute the grid size rule and generate a rule output if necessary.

    Check whether the grid size of (each) kernel launch is at least the number of SMs,
    and generate a rule message and a speedup estimate otherwise.
    In case of range results, find and report the launch with the largest potential speedup.

    Also check if the grid size is less than twice the number of SMs for any launch,
    and suggest to use at least two blocks per SM when __syncthreads() is used.
    """
    rule_name = "Small Grid"
    grid_size_metric = metrics["grid_size"]
    num_sms = int(metrics["device__attribute_multiprocessor_count"].value())

    if grid_size_metric is None:
        # Currently, light range results do not collect launch__grid_size in all cases 
        # and are not supported.
        return

    if grid_size_metric.num_instances() == 0:
        # legacy case, grid size is not instanced per launch
        grid_sizes = [grid_size_metric.value()]
    else:
        grid_sizes = [grid_size_metric.value(i) for i in range(grid_size_metric.num_instances())]

    if any(size < num_sms for size in grid_sizes):
        if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
            grid_size = int(grid_sizes[0])
            workload_specific_part = (
                "The grid for this launch is configured to execute only {:d} {},"
                " which is less than the GPU's {:d} multiprocessors.".format(
                    grid_size, "block" if grid_size == 1 else "blocks", num_sms
                )
            )
            speedup_type, speedup_value = get_estimated_speedup_grid_size(grid_size, num_sms)
        else:
            speedup_type, speedup_value, launch_id, grid_size = \
                get_max_estimated_speedup_grid_size(grid_size_metric, num_sms)
            workload_specific_part = (
                "At least one kernel launch of this workload has a grid which is"
                " configured to execute fewer blocks than the GPU's {:d} multiprocessors"
                " (e.g., {:d} {} for launch ID {:d}).".format(
                    num_sms, grid_size, "block" if grid_size == 1 else "blocks", launch_id
                )
            )

        rule_message = (
            "{} This can underutilize some multiprocessors. If you do not intend to"
            " execute this kernel concurrently with other workloads,"
            " consider reducing the block size to have at least one block per"
            " multiprocessor or increase the size of the grid to fully utilize the"
            " available hardware resources."
            " {}".format(workload_specific_part, HARDWARE_MODEL_REF)
        )

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, rule_message, rule_name)
        fe.speedup(msg_id, speedup_type, speedup_value)
        fe.focus_metric(
            msg_id,
            grid_size_metric.name(),
            grid_size,
            NvRules.IFrontend.Severity_SEVERITY_HIGH,
            (
                "Increase the grid size towards the number of"
                " multiprocessors ({:d})".format(num_sms)
            ),
        )

    # only show the __syncthreads() advice, if we haven't already shown the above advice
    elif any(size < 2 * num_sms for size in grid_sizes):
        if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
            grid_size = int(grid_sizes[0])
            workload_specific_part = (
                "(compared to the currently executed {:.1f} blocks)".format(
                    grid_size / num_sms
                )
            )
        else:
            grid_size = min(grid_sizes)
            instance_id = grid_sizes.index(grid_size)
            launch_id = grid_size_metric.correlation_ids().value(instance_id)
            workload_specific_part = (
                "(compared to the currently executed {:.1f} blocks"
                " for launch ID {:d})".format(grid_size / num_sms, launch_id)
            )

        rule_message = (
            "If you execute __syncthreads() to synchronize the threads of a block,"
            " it is recommended to have at least two blocks per multiprocessor {}"
            " This way, blocks that aren't waiting for __syncthreads()"
            " can keep the hardware busy.".format(workload_specific_part)
        )

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, rule_message, rule_name)
        fe.focus_metric(
            msg_id,
            grid_size_metric.name(),
            grid_size,
            NvRules.IFrontend.Severity_SEVERITY_LOW,
            (
                "Increase the grid size towards twice the number of"
                " multiprocessors ({:d})".format(2 * num_sms)
            ),
        )


def get_problematic_num_waves(num_waves_metric):
    problematic_launches = []

    if num_waves_metric.num_instances() == 0:
        # legacy case, num_waves is not instanced per launch
        num_waves = num_waves_metric.value()
        partial_waves, whole_waves = math.modf(num_waves)
        has_tail_effect = partial_waves > 0 and whole_waves >= 1
        if has_tail_effect:
            instance_id = 0
            problematic_launches.append((instance_id, partial_waves, whole_waves))
        return has_tail_effect, problematic_launches

    for instance_id in range(num_waves_metric.num_instances()):
        num_waves = num_waves_metric.value(instance_id)
        partial_waves, whole_waves = math.modf(num_waves)
        has_tail_effect = partial_waves > 0 and whole_waves >= 1
        if has_tail_effect:
            problematic_launches.append((instance_id, partial_waves, whole_waves))

    return len(problematic_launches) > 0, problematic_launches


def get_estimated_speedup_tail_effect(partial_waves, whole_waves):
    if partial_waves == 0:
        return NvRules.IFrontend.SpeedupType_LOCAL, 0

    improvement_percent = 1 / (1 + whole_waves) * 100
    return NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent


def get_max_estimated_speedup_tail_effect(problematic_launches):
    max_speedup = 0
    launch_id_at_max_speedup = 0
    partial_waves_at_max_speedup = 0
    whole_waves_at_max_speedup = 0

    for launch_id, partial_waves, whole_waves in problematic_launches:
        _, speedup_value = get_estimated_speedup_tail_effect(partial_waves, whole_waves)
        if speedup_value > max_speedup:
            max_speedup = speedup_value
            launch_id_at_max_speedup = launch_id
            partial_waves_at_max_speedup = partial_waves
            whole_waves_at_max_speedup = whole_waves

    # TODO: We could get a global estimate by weighing the speedup by the
    # relative duration of this launch
    return NvRules.IFrontend.SpeedupType_LOCAL, max_speedup, launch_id_at_max_speedup, partial_waves_at_max_speedup, whole_waves_at_max_speedup


def apply_tail_effect_rule(fe, action, metrics):
    """Execute the tail effect rule and generate a rule output if necessary.

    Check whether any kernel launch has a partial wave of thread blocks.
    In case of range results, report the launch with the largest (local) speedup.
    """
    rule_name = "Tail Effect"
    grid_size_metric = metrics["grid_size"]
    num_waves_metric = metrics["num_waves"]

    if num_waves_metric is None:
        # Currently, graph results do not collect launch__waves_per_multiprocessor
        # and are not supported.
        return

    speedup_threshold = 20  # percent

    has_tail_effect, problematic_launches = get_problematic_num_waves(num_waves_metric)

    if has_tail_effect:
        if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
            _, partial_waves, whole_waves = problematic_launches[0]
            grid_size = grid_size_metric.value()
            num_waves = num_waves_metric.value()
            speedup_type, speedup_value = get_estimated_speedup_tail_effect(partial_waves, whole_waves)
            partial_wave_blocks = math.ceil(grid_size * (partial_waves / num_waves))
            workload_specific_part = (
                "This kernel launch results in {:d} full waves and a partial"
                " wave of {:d} thread blocks.".format(
                    int(whole_waves), partial_wave_blocks
                )
            )
        else:
            speedup_type, speedup_value, instance_id, partial_waves, whole_waves = \
                get_max_estimated_speedup_tail_effect(problematic_launches)
            launch_id = num_waves_metric.correlation_ids().value(instance_id)
            grid_size = grid_size_metric.value(instance_id)
            num_waves = num_waves_metric.value(instance_id)
            partial_wave_blocks = math.ceil(grid_size * (partial_waves / num_waves))
            workload_specific_part = (
                "At least one kernel launch of this workload results in a partial wave"
                " (e.g., for launch ID {:d}, {:d} full waves and a partial wave of"
                " {:d} thread blocks are executed).".format(
                    launch_id, int(whole_waves), partial_wave_blocks
                )
            )

        if speedup_value < speedup_threshold:
            return

        rule_message = (
            "A wave of thread blocks is defined as the maximum number of blocks that can"
            " be executed in parallel on the target GPU. The number of blocks in a wave"
            " depends on the number of multiprocessors and the theoretical occupancy"
            " of the kernel. {}"
            " Under the assumption of a uniform execution duration of all thread blocks,"
            " this partial wave may account for up to {:.1f}% of the total runtime of"
            " this kernel."
            " Try launching a grid with no partial wave. The overall impact of this tail"
            " effect also lessens with the number of full waves executed for a grid."
            " {}".format(
                workload_specific_part,
                speedup_value,
                HARDWARE_MODEL_REF,
            )
        )

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, rule_message, rule_name)
        fe.speedup(msg_id, speedup_type, speedup_value)
        fe.focus_metric(
            msg_id,
            num_waves_metric.name(),
            num_waves,
            NvRules.IFrontend.Severity_SEVERITY_DEFAULT,
            (
                "Decrease the number of partial waves"
                " (the fractional part of the number of waves)"
            )
        )


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    apply_block_size_rule(fe, action, metrics)
    apply_grid_size_rule(fe, action, metrics)
    apply_tail_effect_rule(fe, action, metrics)
