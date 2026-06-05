"""
Poem Generation 模型：双编码器 (BiGRU) + Attention Decoder
对应论文 Section 3.3, Figure 2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class KeywordEncoder(nn.Module):
    """关键词编码器：双向 GRU，输出 r_c"""
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=config.PAD_ID)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers,
                         bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: (batch, keyword_len)
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb)
        # outputs: (batch, keyword_len, hidden*2)
        return outputs, hidden


class TextEncoder(nn.Module):
    """前文编码器：双向 GRU"""
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=config.PAD_ID)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers,
                         bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: (batch, text_len)
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau Attention: e_{tj} = v^T tanh(W_a s_{t-1} + U_a h_j)"""
    def __init__(self, hidden_size, attn_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, attn_size, bias=False)
        self.U = nn.Linear(hidden_size * 2, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # decoder_state: (batch, hidden)
        # encoder_outputs: (batch, src_len, hidden*2)
        src_len = encoder_outputs.size(1)
        state_expanded = decoder_state.unsqueeze(1).expand(-1, src_len, -1)
        energy = self.v(torch.tanh(
            self.W(state_expanded) + self.U(encoder_outputs)))
        # energy: (batch, src_len, 1)
        attn_weights = F.softmax(energy.squeeze(2), dim=1)
        # context: (batch, hidden*2)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class AttentionDecoder(nn.Module):
    """Attention Decoder: GRU + Bahdanau Attention"""
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=config.PAD_ID)
        self.attention = BahdanauAttention(hidden_size, hidden_size)
        # input = emb + context
        self.gru = nn.GRU(emb_dim + hidden_size * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + hidden_size * 2, vocab_size)

    def forward_step(self, input_token, hidden, encoder_outputs):
        # input_token: (batch, 1)
        emb = self.embedding(input_token)  # (batch, 1, emb_dim)
        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        # context: (batch, hidden*2)
        gru_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, hidden = self.gru(gru_input, hidden)
        # output projection
        pred = self.fc(torch.cat([output.squeeze(1), context], dim=1))
        return pred, hidden, attn_weights


class PoemGenerationModel(nn.Module):
    """完整模型：双编码器 + Attention Decoder"""
    def __init__(self, vocab_size=config.VOCAB_SIZE, emb_dim=config.EMB_DIM,
                 hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.keyword_encoder = KeywordEncoder(vocab_size, emb_dim, hidden_size, num_layers)
        self.text_encoder = TextEncoder(vocab_size, emb_dim, hidden_size, num_layers)
        self.decoder = AttentionDecoder(vocab_size, emb_dim, hidden_size)
        # 将双向编码器的最终状态映射到 decoder 初始状态
        self.state_proj = nn.Linear(hidden_size * 2, hidden_size)

    def encode(self, keyword_ids, text_ids):
        """编码关键词和前文，返回拼接的 attention states"""
        kw_outputs, kw_hidden = self.keyword_encoder(keyword_ids)
        # kw_outputs: (batch, kw_len, hidden*2) 作为 h_0

        if text_ids is not None and text_ids.size(1) > 0:
            text_outputs, text_hidden = self.text_encoder(text_ids)
            # 拼接: [keyword_outputs ; text_outputs]
            encoder_outputs = torch.cat([kw_outputs, text_outputs], dim=1)
            # decoder 初始状态用 text encoder 最后一层的正向+反向拼接
            last_hidden = torch.cat([text_hidden[-2], text_hidden[-1]], dim=1)
        else:
            encoder_outputs = kw_outputs
            last_hidden = torch.cat([kw_hidden[-2], kw_hidden[-1]], dim=1)

        decoder_hidden = self.state_proj(last_hidden).unsqueeze(0)
        return encoder_outputs, decoder_hidden

    def forward(self, keyword_ids, text_ids, target_ids, teacher_forcing_ratio=1.0):
        """
        keyword_ids: (batch, kw_len)
        text_ids: (batch, text_len) 可以为空 tensor
        target_ids: (batch, target_len) 包含 GO 开头
        """
        batch_size = keyword_ids.size(0)
        target_len = target_ids.size(1)
        encoder_outputs, decoder_hidden = self.encode(keyword_ids, text_ids)

        outputs = torch.zeros(batch_size, target_len - 1, config.VOCAB_SIZE,
                            device=keyword_ids.device)
        input_token = target_ids[:, 0:1]  # GO token

        for t in range(target_len - 1):
            pred, decoder_hidden, _ = self.decoder.forward_step(
                input_token, decoder_hidden, encoder_outputs)
            outputs[:, t] = pred
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = target_ids[:, t+1:t+2]
            else:
                input_token = pred.argmax(dim=1, keepdim=True)

        return outputs

    def generate(self, keyword_ids, text_ids, max_len=config.MAX_LINE_LEN):
        """贪心解码"""
        encoder_outputs, decoder_hidden = self.encode(keyword_ids, text_ids)
        input_token = torch.full((keyword_ids.size(0), 1), config.GO_ID,
                                dtype=torch.long, device=keyword_ids.device)
        generated = []
        for _ in range(max_len):
            pred, decoder_hidden, _ = self.decoder.forward_step(
                input_token, decoder_hidden, encoder_outputs)
            next_token = pred.argmax(dim=1, keepdim=True)
            generated.append(next_token)
            input_token = next_token
        return torch.cat(generated, dim=1)
