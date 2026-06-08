#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum

import torch.nn as nn
from torch.nn.functional import gelu


class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1


TRANSFORMER_VERSION = None

# Alias for BertLayerNorm (removed in modern transformers)
BertLayerNorm = nn.LayerNorm

try:
    from pytorch_pretrained_bert.modeling import (
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertConfig,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead
    )
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'
    TRANSFORMER_VERSION = TransformerVersion.PYTORCH_PRETRAINED_BERT
    logging.warning('You are using the old version of `pytorch_pretrained_bert`')
except ImportError:
    from transformers import BertTokenizer, BertConfig  # noqa
    from transformers.models.bert.modeling_bert import (    # noqa
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead
    )

    hf_flag = 'new'
    TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS
