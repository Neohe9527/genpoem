"""Encoder-Decoder Transformer: query → 整首诗"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoemEncoderDecoder(nn.Module):
    def __init__(self, vocab_size=6000, d_model=512, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=6,
                 dim_ff=1024, dropout=0.15, tie_weights=True, use_segment=True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, d_model) if use_segment else None
        self.pos_enc = SinusoidalPE(d_model, dropout, max_len=64)
        self.pos_dec = SinusoidalPE(d_model, dropout, max_len=48)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers,
                                            enable_nested_tensor=False)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.tie_weights = tie_weights
        if tie_weights:
            self.fc_out.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.embedding.weight.data[0].zero_()
        if self.segment_embedding is not None:
            nn.init.normal_(self.segment_embedding.weight, std=0.02)

    def encode(self, src, src_pad_mask=None):
        x = self.embedding(src)
        if self.segment_embedding is not None:
            seg = torch.zeros_like(src)
            seg[:, 1:] = 1  # first token is format token
            x = x + self.segment_embedding(seg)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_pad_mask=None):
        x = self.embedding(tgt)
        x = self.pos_dec(x)
        x = self.decoder(x, memory, tgt_mask=tgt_mask,
                         memory_key_padding_mask=memory_pad_mask)
        if self.tie_weights:
            return self.fc_out(x)
        return self.fc_out(x)

    def forward(self, src, tgt, src_pad_mask=None):
        """Training forward: src=query_ids, tgt=poem_ids (with GO prepended)"""
        memory = self.encode(src, src_pad_mask)
        tgt_input = tgt[:, :-1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_input.size(1), device=src.device)
        logits = self.decode(tgt_input, memory, tgt_mask=tgt_mask,
                            memory_pad_mask=src_pad_mask)
        return logits

    def generate(self, src, max_len=35, temperature=0.75, top_p=0.9,
                 repetition_penalty=1.3, go_id=1, eos_id=2, pad_id=0):
        """Autoregressive generation"""
        src_pad_mask = (src == pad_id)
        memory = self.encode(src, src_pad_mask)
        batch_size = src.size(0)
        generated = torch.full((batch_size, 1), go_id,
                              dtype=torch.long, device=src.device)

        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1), device=src.device)
            logits = self.decode(generated, memory, tgt_mask=tgt_mask,
                               memory_pad_mask=src_pad_mask)
            logits = logits[:, -1] / temperature

            # Repetition penalty (within this poem)
            for b in range(batch_size):
                seen = generated[b, 1:].tolist()
                for tid in set(seen):
                    if tid > 4:
                        logits[b, tid] /= repetition_penalty
                # Ban consecutive repeat
                if len(seen) > 0 and seen[-1] > 4:
                    logits[b, seen[-1]] = float('-inf')

            # Block PAD/GO/UNK but allow SEP and EOS
            logits[:, pad_id] = float('-inf')
            logits[:, go_id] = float('-inf')
            logits[:, 3] = float('-inf')  # UNK

            probs = F.softmax(logits, dim=-1)

            # Nucleus sampling
            if top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences produced EOS
            if (next_token == eos_id).all():
                break

        return generated[:, 1:]  # Remove GO


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
