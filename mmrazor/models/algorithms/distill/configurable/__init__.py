# Copyright (c) OpenMMLab. All rights reserved.
from .datafree_distillation import (DAFLDataFreeDistillation,
                                    DataFreeDistillation)
from .fpn_teacher_distill import FpnTeacherDistill
from .overhaul_feature_distillation import OverhaulFeatureDistillation
from .self_distill import SelfDistill
from .single_teacher_distill import SingleTeacherDistill
from .custom_distill import CustomDistill

__all__ = [
    'SelfDistill', 'SingleTeacherDistill', 'FpnTeacherDistill',
    'DataFreeDistillation', 'DAFLDataFreeDistillation',
    'OverhaulFeatureDistillation', 'CustomDistill'
]
