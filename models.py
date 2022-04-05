# Copyright 2022 Informatix Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from transformers import RobertaModel
import torch
from torch import nn


class IFXRoberta(nn.Module):
    def __init__(self, roberta: RobertaModel, vocab_size: int = 32005, model_dim: int = 768):
        super().__init__()
        self.roberta = roberta
        self.linear_word = nn.Linear(model_dim, vocab_size)

    def forward(self, x: torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        out = self.roberta(x, attention_mask=attention_mask)
        x = out.last_hidden_state
        return self.masked_lm(x)

    def masked_lm(self, x: torch.FloatTensor) -> torch.Tensor:
        return torch.log_softmax(self.linear_word(x), dim=-1)
