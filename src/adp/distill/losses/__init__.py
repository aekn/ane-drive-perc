from adp.distill.losses.box import BoxDistillLoss
from adp.distill.losses.cls import ClsDistillLoss
from adp.distill.losses.fdr import FDRDistributionDistillLoss
from adp.distill.losses.feature import AIFIFeatureDistillLoss
from adp.distill.losses.matching import gt_mediated_pairs
from adp.distill.losses.utils import aifi_ramp_schedule, compute_pair_quality_weights


__all__ = [
    "BoxDistillLoss",
    "ClsDistillLoss",
    "FDRDistributionDistillLoss",
    "AIFIFeatureDistillLoss",
    "gt_mediated_pairs",
    "compute_pair_quality_weights",
    "aifi_ramp_schedule",
]
