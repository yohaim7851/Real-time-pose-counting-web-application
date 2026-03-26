"""
PoseRAC — Transformer 기반 포즈 분류 모델 (추론 전용)

입력: (batch, 104) 포즈 피처 벡터
출력: (batch, num_classes) 클래스 확률 (sigmoid 적용 전)
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn

torch.multiprocessing.set_sharing_strategy("file_system")


class PoseRAC(pl.LightningModule):
    """
    Transformer Encoder + FC 레이어로 구성된 포즈 분류 모델.
    추론(forward)만 사용합니다.
    """

    def __init__(
        self,
        dim: int, heads: int, enc_layer: int,
        learning_rate: float, seed: int,
        num_classes: int, alpha: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads),
            num_layers=enc_layer,
        )
        self.fc1         = nn.Linear(dim, num_classes)
        self.dim         = dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.alpha         = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.dim)
        x = self.transformer_encoder(x)
        x = x.view(-1, self.dim)
        return self.fc1(x)


class ActionTrigger:
    """
    히스테리시스 기반 렙 카운팅 트리거.

    동작:
      - 점수가 enter_threshold 초과 → 동작 진입 상태로 전환
      - 진입 상태에서 exit_threshold 미만 → 트리거 발동 (렙 +1)
    """

    def __init__(
        self,
        action_name: str,
        enter_threshold: float = 0.8,
        exit_threshold: float = 0.4,
    ):
        self._action_name     = action_name
        self._enter_threshold = enter_threshold
        self._exit_threshold  = exit_threshold
        self._pose_entered    = False

    def __call__(self, pose_score: float) -> bool:
        if not self._pose_entered:
            self._pose_entered = pose_score > self._enter_threshold
            return False
        if pose_score < self._exit_threshold:
            self._pose_entered = False
            return True
        return False

