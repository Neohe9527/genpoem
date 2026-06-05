"""
Transformer-based Poem Generation Model
替换 BiGRU Encoder-Decoder，保持相同接口
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PoemTransformer(nn.Module):
    def __init__(self, vocab_size=config.VOCAB_SIZE, d_model=256, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3, dim_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=64)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def encode(self, src, src_key_padding_mask=None):
        x = self.embedding(src) * self.scale
        x = self.pos_encoder(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        x = self.embedding(tgt) * self.scale
        x = self.pos_decoder(x)
        x = self.decoder(x, memory, tgt_mask=tgt_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(x)

    def forward(self, keyword_ids, text_ids, target_ids, teacher_forcing_ratio=1.0):
        # Concat keyword + text as encoder input
        src = torch.cat([keyword_ids, text_ids], dim=1)
        src_pad_mask = (src == config.PAD_ID)

        memory = self.encode(src, src_key_padding_mask=src_pad_mask)

        # Target: exclude last token for input, exclude first (GO) for label
        tgt_input = target_ids[:, :-1]
        tgt_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src.device)

        logits = self.decode(tgt_input, memory, tgt_mask=tgt_mask,
                            memory_key_padding_mask=src_pad_mask)
        return logits

    def generate(self, keyword_ids, text_ids, max_len=config.MAX_LINE_LEN):
        src = torch.cat([keyword_ids, text_ids], dim=1)
        src_pad_mask = (src == config.PAD_ID)
        memory = self.encode(src, src_key_padding_mask=src_pad_mask)

        batch_size = keyword_ids.size(0)
        generated = torch.full((batch_size, 1), config.GO_ID,
                              dtype=torch.long, device=src.device)

        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1), device=src.device)
            logits = self.decode(generated, memory, tgt_mask=tgt_mask,
                               memory_key_padding_mask=src_pad_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Alias for compatibility with generate.py
PoemGenerationModel = PoemTransformer
