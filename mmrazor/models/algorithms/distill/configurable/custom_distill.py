# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.structures import BaseDataElement

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS
from ...base import LossResults
from .single_teacher_distill import SingleTeacherDistill


@MODELS.register_module()
class CustomDistill(SingleTeacherDistill):
    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        losses = dict()

        self.distiller.set_deliveries_override(False)
        if self.teacher_trainable:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                # _ = self.teacher.extract_feat(batch_inputs)
                _ = self.teacher._forward(batch_inputs)
        else:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                with torch.no_grad():
                    # _ = self.teacher.extract_feat(batch_inputs)
                    _ = self.teacher._forward(batch_inputs)

        self.distiller.set_deliveries_override(True)
        with self.distiller.student_recorders, self.distiller.deliveries:
            student_losses = self.student(batch_inputs, data_samples, mode='loss')
        losses.update(add_prefix(student_losses, 'student'))

        if not self.distillation_stopped:
            distill_losses = self.distiller.compute_distill_losses()
            losses.update(add_prefix(distill_losses, 'distill'))

        return losses
