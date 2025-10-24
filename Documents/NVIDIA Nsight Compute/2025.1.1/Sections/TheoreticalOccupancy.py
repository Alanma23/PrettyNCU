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
from enum import StrEnum, auto

import NvRules
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser

requested_metrics = [
    MetricRequest("smsp__maximum_warps_avg_per_active_cycle", "theoretical_warps", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__warps_active.avg.peak_sustained", "max_warps", Importance.OPTIONAL, None, False),
    MetricRequest("launch__occupancy_limit_blocks", None, Importance.OPTIONAL, None, False),
    MetricRequest("launch__occupancy_limit_registers", None, Importance.OPTIONAL, None, False),
    MetricRequest("launch__occupancy_limit_shared_mem", None, Importance.OPTIONAL, None, False),
    MetricRequest("launch__occupancy_limit_warps", None, Importance.OPTIONAL, None, False),
]


class LIMIT_TYPES(StrEnum):
    blocks = auto()
    registers = auto()
    shared_mem = auto()
    warps = auto()


def get_identifier():
    return "TheoreticalOccupancy"


def get_name():
    return "Theoretical Occupancy"


def get_description():
    return "Analysis of Theoretical Occupancy and its Limiters"


def get_section_identifier():
    return "Occupancy"


def get_parent_rules_identifiers():
    return ["IssueSlotUtilization"]


def get_top_limiters(metrics):
    limiters = []

    for limiter in LIMIT_TYPES:
        limit_value = metrics[f"launch__occupancy_limit_{limiter}"].value()
        limiters.append((limiter, limit_value))

    # Get the limiter(s) with the lowest value (as they are most restrictive)
    limiters.sort(key=lambda limit: limit[1])
    top_limiters = [limiters[0][0]]
    top_value = limiters[0][1]
    index = 1

    while index < len(limiters) and limiters[index][1] == top_value:
        top_limiters.append(limiters[index][0])
        index += 1

    return top_limiters


def get_estimated_speedup(parent_weights, theoretical_warps, max_warps):
    improvement_local = 1 - theoretical_warps / max_warps

    parent_speedup_name = "issue_slot_util_speedup_normalized"
    if parent_speedup_name in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        improvement_global = min(parent_weights[parent_speedup_name], improvement_local)
        improvement_percent = improvement_global * 100
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
        improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent


def get_max_estimated_speedup(parent_weights, theoretical_warps_metric, max_warps):
    if theoretical_warps_metric.num_instances() == 0:
        theoretical_warps = theoretical_warps_metric.value()
        launch_id = 0
        speedup_type, speedup_value = get_estimated_speedup(
            parent_weights, theoretical_warps, max_warps
        )
        return speedup_type, speedup_value, launch_id, theoretical_warps

    max_speedup = 0
    launch_id_at_max_speedup = 0
    theoretical_warps_at_max_speedup = 0

    for instance_id in range(theoretical_warps_metric.num_instances()):
        theoretical_warps = theoretical_warps_metric.value(instance_id)
        _, speedup_value = get_estimated_speedup(
            parent_weights, theoretical_warps, max_warps
        )

        if speedup_value > max_speedup:
            max_speedup = speedup_value
            launch_id_at_max_speedup = \
                theoretical_warps_metric.correlation_ids().value(instance_id)
            theoretical_warps_at_max_speedup = theoretical_warps

    # TODO: We could get a global estimate by forming the weighted average of all
    #       speedups, weighed by the relative duration of each launch.
    return NvRules.IFrontend.SpeedupType_LOCAL, max_speedup, launch_id_at_max_speedup, theoretical_warps_at_max_speedup


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    if any(metric is None for metric in metrics.values()):
        # Not all occupancy metrics are available for all supported workload/profile mode
        # combinations, so we skip the rule if any of them are missing.
        return

    parent_weights = fe.receive_dict_from_parent("IssueSlotUtilization")

    description_limit = {
        LIMIT_TYPES.blocks: "the number of blocks that can fit on the SM",
        LIMIT_TYPES.registers: "the number of required registers",
        LIMIT_TYPES.shared_mem: "the required amount of shared memory",
        LIMIT_TYPES.warps: "the number of warps within each block",
    }

    theoretical_warps_metric = metrics["theoretical_warps"]
    max_warps = int(metrics["max_warps"].value())

    low_theoretical_threshold = 80

    if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
        theoretical_warps = theoretical_warps_metric.value()
        speedup_type, speedup_value = get_estimated_speedup(
            parent_weights, theoretical_warps, max_warps
        )
    else:
        speedup_type, speedup_value, launch_id, theoretical_warps = \
            get_max_estimated_speedup(parent_weights, theoretical_warps_metric, max_warps)

    theoretical_warps_pct_of_peak = (theoretical_warps / max_warps) * 100
    if theoretical_warps_pct_of_peak >= low_theoretical_threshold:
        return

    top_limiters = get_top_limiters(metrics)
    top_limiters_string = \
        description_limit[top_limiters[0]] if len(top_limiters) == 1 \
        else description_limit[top_limiters[0]] + ", ".join(
            [description_limit[limiter] for limiter in top_limiters[1:-2]]
        ) + ", and " + description_limit[top_limiters[-1]]
    top_limiters_message = (
        "This kernel's theoretical occupancy ({:.1f}%) is limited by {}.".format(
            theoretical_warps_pct_of_peak, top_limiters_string
        )
    )

    if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
        message = (
            "The {:.2f} theoretical warps per scheduler this kernel can issue according"
            " to its occupancy are below the hardware maximum of {}. {}".format(
                theoretical_warps, max_warps, top_limiters_message
            )
        )
    else:
        message = (
            "For some launches of this workload, the theoretical number of warps per"
            " scheduler that can be issued according to its occupancy are below the"
            " hardware maximum of {}, e.g., for the kernel with launch ID {}."
            " {}".format(
                max_warps, launch_id, top_limiters_message
            )
        )

    msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)
    fe.speedup(msg_id, speedup_type, speedup_value)
    fe.focus_metric(
        msg_id,
        metrics["theoretical_warps"].name(),
        theoretical_warps,
        NvRules.IFrontend.Severity_SEVERITY_HIGH,
        "Increase the theoretical number of warps per scheduler that can be issued",
    )
