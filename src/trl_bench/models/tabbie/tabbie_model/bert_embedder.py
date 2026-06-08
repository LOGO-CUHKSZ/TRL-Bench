"""
BERT Token Embedder for TABBIE

Patched from: table_embedder/models/lib/bert_token_embedder.py

Replacements:
  - pytorch_pretrained_bert.modeling.BertModel → transformers.BertModel
  - Old API: (all_encoder_layers, pooled) → New API: .last_hidden_state
  - Removed NVIDIA Apex .half() (use float32 for inference)
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertEmbedder(nn.Module):
    """Wraps HuggingFace BertModel to produce per-token embeddings.

    The original TABBIE used pytorch_pretrained_bert which returns
    (all_encoder_layers, pooled_output). The new transformers API
    returns a BaseModelOutputWithPoolingAndCrossAttentions object
    with .last_hidden_state.
    """

    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1=real token, 0=padding

        Returns:
            (batch, seq_len, 768) last hidden state
        """
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
