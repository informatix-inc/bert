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

import csv
import re
import sys
import traceback
import unicodedata

import mojimoji
import numpy as np
from pyknp import Juman
from subword_nmt.apply_bpe import BPE
from typing import List, Optional


class JumanSeparator:
    def __init__(self, juman_conf_file: Optional[str], period: Optional[str] = "。"):
        assert period is None or type(period) is str
        if juman_conf_file:
            self.jumanpp = Juman(option="-c {}".format(juman_conf_file))
        else:
            self.jumanpp = Juman()
        space_chars = "".join(chr(c) for c in range(sys.maxunicode + 1))
        self.white_space = "|".join(re.findall(r"\s", space_chars))
        self.period = period

    def __call__(self, doc: str) -> List[List[str]]:
        separate_list = []
        lines = doc.split("\n")
        for l in lines:
            splits = l.split(self.period) if self.period else [l]
            for spl in splits:
                sentence = spl.strip()
                if not sentence:
                    continue
                if self.period:
                    sentence += self.period
                # normalize and clean senetence
                sentence = unicodedata.normalize("NFKC", sentence)
                sentence = re.sub(self.white_space, "", sentence)
                sentence = mojimoji.han_to_zen(sentence)

                try:
                    mlist = self.jumanpp.analysis(sentence)
                except Exception:
                    print(traceback.format_exc())
                    continue

                mlist = [mrph.midasi for mrph in mlist.mrph_list()]
                separate_list.append(mlist)

        return separate_list


class BPEClassifier:
    def __init__(self, bpe_file: str, count_file: str, sep_cls: int = 0, init_cls: int = 3, unknown_cls: int = 4):
        # load file for "byte pair encoding"
        with open(bpe_file, "r", encoding="utf_8", errors="ignore") as f:
            self.bpe = BPE(f)

        # load file of "correspondence map between word piece and class number"
        with open(count_file, "r", encoding="utf_8", errors="ignore") as f:
            count_data = csv.reader(f)
            self.clist = [d[1] for d in count_data]

        self.sep_cls = sep_cls
        self.init_cls = init_cls
        self.unknown_cls = unknown_cls

    def __call__(self, token_list: List[List[str]]) -> List[int]:
        doc_cls = [self.init_cls]

        for tlist in token_list:
            # skip short sentence
            if len(tlist) < 2:
                continue
            segment_result = self.bpe.segment_tokens(tlist)
            result_cls = [self.clist.index(w) if w in self.clist else self.unknown_cls for w in segment_result]
            doc_cls.extend(result_cls)
            doc_cls.append(self.sep_cls)

        return doc_cls


class TextProcessor:
    def __init__(self, bpe_file: str, count_file: str, juman_conf_file: Optional[str] = None):
        """
        Encodes Japanese text into list of predefined class numbers, and decodes encoded text into list of word pieces.
        Args:
            bpe_file: Path to file which defines word pieces for byte pair encoding. For more detail, please refer to subword_nmt documentation.
            count_file: Path to file which defines correspondence between word piece and class number.
            juman_conf_file(optional): Path to config file for Juman. For more detail, please refer to Juman documentation.
        """
        self.juman = JumanSeparator(juman_conf_file=juman_conf_file)
        self.bpe = BPEClassifier(bpe_file=bpe_file, count_file=count_file)

    def encode(self, text: str) -> List[int]:
        """
        Encodes Japanese sentences into list of integers, which is predefined class numbers.
        Args:
            text: Japanese sentences.
        Returns:
            List of encoded word pieces, including special tokens.
        """
        sep_text = self.juman(text)
        enc_text = self.bpe(sep_text)
        return enc_text

    def decode(self, ids: List[int], ignore_idx: List[int] = []) -> List[str]:
        """
        Decodes a list of encoded word pieces into a list of word pieces.
        Args:
            ids: List of encoded word pieces.
            ignore_idx: List of index to be ignored.
        Returns:
            List of word pieces.
        """
        return [self.bpe.clist[c] for i, c in enumerate(ids) if i not in ignore_idx]

    def random_mask(self, ids: List[int], mask_rate: float = 0.15, mask_cls: int = 1) -> List[int]:
        """
        Randomly masks a list of encoded word pieces.
        Args:
            ids: List of encoded word pieces.
            mask_rate: Rate of replacement.
            mask_cls: Class number of mask tokens.
        Returns:
            List of randomly masked tokens.
        """
        ids = np.array(ids, dtype=np.int)
        target_idx = np.where(ids >= 5)[0]  # Avoid masking special tokens
        mask_idx = np.random.choice(target_idx, size=int(mask_rate * len(target_idx)), replace=False)
        ids[mask_idx] = mask_cls
        ids = ids.tolist()
        return ids
