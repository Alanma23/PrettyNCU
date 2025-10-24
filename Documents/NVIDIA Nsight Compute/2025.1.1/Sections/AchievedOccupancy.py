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
import NvRules
from RequestedMetrics import Importance, MetricRequest, RequestedMetricsParser

requested_metrics = [
    MetricRequest("sm__maximum_warps_per_active_cycle_pct", "theoretical_occupancy", Importance.OPTIONAL, None, False),
    MetricRequest("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved_occupancy", Importance.OPTIONAL, None, False),
]


def get_identifier():
    return "AchievedOccupancy"

def get_name():
    return "Achieved Occupancy"

def get_description():
    return "Analysis of the Achieved Occupancy"

def get_section_identifier():
    return "Occupancy"

def get_parent_rules_identifiers():
    return ["IssueSlotUtilization"]


def get_estimated_speedup(parent_weights, metrics):
    """Estimate potential speedup from increasing the achieved occupancy.

    The performance improvement is approximated as relative proportion of the difference
    of theoretical and achieved occupancy.
    In case it's available, the performance improvement can be upper-bounded by the
    speedup estimate of IssueSlotUtilization.

    """
    theoretical_occupancy = metrics["theoretical_occupancy"].value()
    achieved_occupancy = metrics["achieved_occupancy"].value()
    improvement_local = (theoretical_occupancy - achieved_occupancy) / theoretical_occupancy

    parent_speedup_name = "issue_slot_util_speedup_normalized"
    if parent_speedup_name in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        improvement_global = min(
            parent_weights[parent_speedup_name], improvement_local
        )
        improvement_percent = improvement_global * 100
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
        improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent


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

    load_imbalance_advice = (
        "Load imbalances can occur between warps within a block as well as across"
        " blocks of the same kernel."
        " See the @url:CUDA Best Practices"
        " Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@"
        " for more details on optimizing occupancy."
    )

    occupancy_difference_threshold = 10  # percent

    if action.workload_type() == NvRules.IAction.WorkloadType_KERNEL:
        theoretical_occupancy = metrics["theoretical_occupancy"].value()
        achieved_occupancy = metrics["achieved_occupancy"].value()

        occupancy_difference = theoretical_occupancy - achieved_occupancy
        if occupancy_difference <= occupancy_difference_threshold:
            return

        message = (
            "The difference between calculated theoretical ({:.1f}%) and measured"
            " achieved occupancy ({:.1f}%) can be the result of warp scheduling overheads"
            " or workload imbalances during the kernel execution."
            " {}".format(
                theoretical_occupancy, achieved_occupancy, load_imbalance_advice
            )
        )

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)

        speedup_type, speedup_value = get_estimated_speedup(parent_weights, metrics)
        fe.speedup(msg_id, speedup_type, speedup_value)

        fe.focus_metric(msg_id, metrics["achieved_occupancy"].name(), achieved_occupancy, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Increase the achieved occupancy towards the theoretical limit ({:.1f}%)".format(theoretical_occupancy))

    else:
        theoretical_occupancy_metric = metrics["theoretical_occupancy"]  # per launch
        achieved_occupancy = metrics["achieved_occupancy"].value()  # aggregated over workload

        low_occupancy_launches = []
        for instance_id in range(theoretical_occupancy_metric.num_instances()):
            theoretical_occupancy = theoretical_occupancy_metric.value(instance_id)
            occupancy_difference = theoretical_occupancy - achieved_occupancy

            # This also excludes launches with theoretical occupancy < achieved occupancy;
            # we cannot provide meaningful advice for such cases.
            if occupancy_difference > occupancy_difference_threshold:
                launch_id = theoretical_occupancy_metric.correlation_ids().value(instance_id)
                low_occupancy_launches.append((launch_id, theoretical_occupancy))

        if len(low_occupancy_launches) == 0:
            return

        # Report up to 3 launches with the largest gaps to the average achieved occupancy
        low_occupancy_launches.sort(reverse=True, key=lambda x: x[1])
        message = (
            "The large difference between the calculated theoretical occupancy (per launch)"
            " and the measured achieved occupancy (of the entire workload) for some"
            " launches of this workload (e.g., {}) can be the result of warp scheduling"
            " overheads or workload imbalances during the kernel execution. {}".format(
                "launch ID {}".format(low_occupancy_launches[0][0]) if len(low_occupancy_launches) == 1
                else "launch IDs {}".format(", ".join(str(launch[0]) for launch in low_occupancy_launches[:3])),
                load_imbalance_advice,
            )
        )

        # NOTE: Currently, we cannot estimate a speedup without having the achieved
        #       occupancy per launch (for local speedups), as well as the runtime of
        #       individual launches (for global speedups).
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)
