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

import json

import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel
import yaml

from tokenizer import TextProcessor
from models import IFXRoberta

if __name__ == "__main__":
    # load config file
    with open("config.yaml") as f:
        conf = yaml.safe_load(f)
    bpe_file = conf["file_paths"]["bpe_file"]
    count_file = conf["file_paths"]["count_file"]
    roberta_weight_path = conf["file_paths"]["roberta_weight"]
    linear_weight_path = conf["file_paths"]["linear_weight"]
    roberta_config_path = conf["file_paths"]["roberta_config"]
    juman_config_path = conf["file_paths"]["juman_config"]
    device_type = conf["device_type"]

    device = torch.device(device_type)

    # load tokenizer
    processor = TextProcessor(bpe_file=bpe_file, count_file=count_file, juman_conf_file=juman_config_path)

    # load pretrained bert model
    with open(roberta_config_path, "r") as f:
        config_dict = json.load(f)
    config_bert = RobertaConfig().from_dict(config_dict)
    roberta = RobertaModel(config=config_bert)
    roberta.load_state_dict(torch.load(roberta_weight_path, map_location=device))

    # load pretained decoder
    ifxroberta = IFXRoberta(roberta)
    ifxroberta.linear_word.load_state_dict(torch.load(linear_weight_path, map_location=device))
    ifxroberta.eval()

    # infer
    rinen1 = "世の中に己の能力をもって奉仕することにより、己の価値を問い、己の存在を主張し、そしてまた、世の中から求められる己の存在価値を創造する。"
    rinen2 =  "コンピュータ技術およびその関連技術に対する研究開発に努め、自己研鑽に励み、専門的な技術集団を形成する。"
    rinen3 = "人間関係が常に仕事の基本にあることを認識し、仕事を通じて、より建設的な人間関係を形成することを追求する。"
    inp_text = rinen1 + rinen2 + rinen3
    bpe_ids = processor.encode(inp_text)
    masked_ids = processor.random_mask(bpe_ids)
    with torch.no_grad():
        inp_tensor = torch.LongTensor([masked_ids])
        inp_tensor.to(device)
        out_tensor = ifxroberta(inp_tensor)

    # decoding masked text and model output
    out_text_code = torch.max(out_tensor, dim=-1, keepdim=True)[1][0]
    ids = out_text_code.squeeze(-1).cpu().numpy().tolist()
    ignore_idx = np.where(np.array(bpe_ids) < 5)[0]
    out_text = processor.decode(ids, ignore_idx=ignore_idx)
    masked_text = processor.decode(masked_ids)
    print("原文:", inp_text)
    print("入力:", "".join(masked_text).replace("@@", ""))
    print("出力:", "".join(out_text).replace("@@", ""))
