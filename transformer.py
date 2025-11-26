import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads: int, embedding_dim: int):
        super().__init__()
        assert embedding_dim % n_heads == 0
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // n_heads
        self.transform_QKV = nn.Linear(embedding_dim, 3*embedding_dim, bias=False)
        self.linear_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.transform_QKV.weight)
        nn.init.xavier_uniform_(self.linear_layer.weight)

    def forward(self, x: tc.Tensor, mask: Optional[tc.Tensor]=None):
        qkv = self.transform_QKV(x) # shape (batch, seq_len, 3*embedding_dim)
        batch_dim, seq_len, embedding_dim = x.shape
        qkv = qkv.reshape(batch_dim, seq_len, self.n_heads, 3*self.head_dim).transpose(1,2) # shape (batch, head, seq_len, 3*head_dim)
        q, k, v = tc.chunk(qkv, 3, -1) # shape (batch, head, seq_len, head_dim)
        attn = self.scaled_dot_product_attention(q, k, v, mask=mask)  # shape (batch, head, seq_len, head_dim)
        attn = attn.transpose(1,2).reshape(batch_dim, seq_len, embedding_dim)  # (shape batch, seq_len, embedding_dim)
        out = self.linear_layer(attn)
        return out

    @staticmethod
    def scaled_dot_product_attention(q: tc.Tensor, k: tc.Tensor, v: tc.Tensor, mask: Optional[tc.Tensor]=None):
        a = MultiHeadAttention.attention_map(q, k, mask)
        values = tc.einsum('...tu,...uh->...th', a, v)
        return values
    
    @staticmethod
    def attention_map(q: tc.Tensor, k: tc.Tensor, mask: Optional[tc.Tensor]=None):
        a = tc.einsum('...th,...uh->...tu', q, k)
        a = a / math.sqrt(k.shape[-1])
        if mask is not None:
            a.masked_fill(mask==0, -tc.inf)
        amaps = F.softmax(a, dim=-1)
        return amaps

class EncoderBlock(nn.Module):

    def __init__(self, n_heads: int, embedding_dim: int, hidden_dim: int, dropout_p: float=0.5):
        super().__init__()
        self.attention_block = MultiHeadAttention(n_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.MLP = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.Dropout(dropout_p),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, embedding_dim))
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: tc.Tensor, mask: Optional[tc.Tensor]=None):
        attn = self.attention_block(x, mask)
        x = x + self.dropout(attn)
        x = self.norm1(x)
        lin = self.MLP(x)
        x = x + self.dropout(lin)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers: int, n_heads: int, embedding_dim: int, hidden_dim: int, droupout_p: float=0.5):
        super().__init__()
        self.encoder_stack = nn.ModuleList([EncoderBlock(n_heads, embedding_dim, hidden_dim, droupout_p)
                                            for _ in range(n_layers)])
    
    def forward(self, x: tc.Tensor, mask: Optional[tc.Tensor]=None):
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x, mask)
        return x
    
    def get_attention_maps(self, x: tc.Tensor, mask: Optional[tc.Tensor]=None):
        attention_maps = []
        for layer in self.encoder_stack:
            attn_map = layer.attention_map(x, mask=mask)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000):
        super().__init__()
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_seq_len inputs
        pe = tc.zeros(max_seq_len, embedding_dim)
        position = tc.arange(0, max_seq_len, dtype=tc.float).unsqueeze(1)
        div_term = tc.exp(tc.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = tc.sin(position * div_term)
        pe[:, 1::2] = tc.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
    
class ObservationModel(nn.Module):
    def __init__(self, obs_dim, embedding_dim):
        super().__init__()
        self.layer = nn.Linear(embedding_dim, obs_dim, bias=False)
    
    def forward(self, z):
        return self.layer(z)
    
    def embed(self, x):
        inv = tc.pinverse(self.layer.weight)
        return tc.einsum('ij,...j->...i', inv, x)


class Transformer(nn.Module):

    def __init__(self, obs_dim, embedding_dim, hidden_dim, n_heads, n_layers, dropout_p=0.5):
        super().__init__()
        self.obs_model = ObservationModel(obs_dim, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len=1000)
        self.transformer_encoder = TransformerEncoder(n_layers, n_heads, embedding_dim, hidden_dim, dropout_p)
    
    def forward(self, x: tc.Tensor, add_positional_encoding: bool=True):
        z = self.obs_model.embed(x)
        if add_positional_encoding:
            z = self.pos_encoder(z)
        z = self.transformer_encoder(z)
        x_hat = self.obs_model(z)
        return x_hat
    
    @tc.no_grad()
    def get_attention_maps(self, x: tc.Tensor, mask: Optional[tc.Tensor]=None, add_positional_encoding: bool=True):
        z = self.obs_model.embed(x)
        if add_positional_encoding:
            z = self.pos_encoder(z)
        attention_maps = self.transformer_encoder.get_attention_maps(x, mask=mask)
        return attention_maps