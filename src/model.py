import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from typing import Optional, Dict
# from __future__ import annotations
# import omegaconf
from inspect import signature ## modern bert에서 token_type_ids 키워드 전달 경우

class EncoderForClassification(nn.Module):
    def __init__(self, model_config): # model_config : omegaconf.DictConfig
        super().__init__()

        # 1) 백본 구성/로드
        self.model_name: str = model_config.model_name
        self.backbone = AutoModel.from_pretrained(self.model_name)
        self.hidden_size: int = self.backbone.config.hidden_size

        # 2) 하이퍼/옵션
        self.pooling: str = getattr(model_config, "pooling", "cls")
        self.dropout = nn.Dropout(float(getattr(model_config, "dropout", 0.1)))
        num_labels = int(getattr(model_config, "num_labels", 2))

        # 3) 분류 헤드
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self._init_classifier()

        # freeze 옵션???
        #self.freeze_backbone = bool(getattr(model_config, "freeze_backbone", False))
        #if self.freeze_backbone:
        #    for p in self.backbone.parameters():
        #        p.requires_grad = False
        #    # (선택) 배치정규/드롭아웃 영향 최소화
        #    self.backbone.eval()
        #raise NotImplementedError

    def _init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    def _pool(self, outputs, attention_mask:torch.Tensor) -> torch.Tensor:
        """
        은닉 상태에서 문장 단위 표현을 만들기 위한 풀링 함수
        Inputs:
            outputs        : AutoModel의 출력 (BaseModelOutputWithPooling 등)
            attention_mask : LongTensor (B, T)

        Returns:
            sent_rep : FloatTensor (B, hidden_size)
        """
        if self.pooling == 'cls':
            # 1. pooler_output이 있는 모델 (BERT)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output # (B,H)
            # 2. 없는 모델은 첫 토큰(hidden_state[:, 0]) 사용
            return outputs.last_hidden_state[:,0]
        elif self.pooling == 'mean':
            # attention_mask로 가려진 토큰 평균
            # last_hidden_state: (B, T, H)
            x = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).type_as(x)  # (B, T, 1)
            summed = (x * mask).sum(dim=1)                  # (B, H)
            denom = mask.sum(dim=1).clamp(min=1e-6)         # (B, 1)
            return summed / denom
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, 
                input_ids : torch.Tensor, 
                attention_mask : torch.Tensor, 
                token_type_ids: Optional[torch.Tensor] = None,
                label: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        사전학습 인코더 → 풀링 → 드롭아웃 → Linear → 로짓/로스

        Inputs:
            input_ids      : LongTensor (B, T)
            attention_mask : LongTensor (B, T)
            token_type_ids : LongTensor (B, T) | None
            label          : LongTensor (B,) | None

        Outputs:
            {
              'logits': FloatTensor (B, num_labels),
              'loss'  : FloatTensor (,) | None
            }
        """

        ## 일부 모델은 token_type_ids를 사용하지 않음 → None 전달하면 내부에서 무시
        #outputs = self.backbone(
        #    input_ids=input_ids,
        #    attention_mask=attention_mask,
        #    token_type_ids=token_type_ids if token_type_ids is not None else None,
        #)
        try:
            outputs = self.backbone(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        except TypeError:
            outputs = self.backbone(input_ids=input_ids,
                                    attention_mask=attention_mask)


        sent_rep = self._pool(outputs, attention_mask)     # (B, H)
        logits = self.classifier(self.dropout(sent_rep))   # (B, C)

        loss = None
        if label is not None:
            # CrossEntropyLoss는 target이 LongTensor 필요
            loss = F.cross_entropy(logits, label)

        return {"logits": logits, "loss": loss}
        #raise NotImplementedError