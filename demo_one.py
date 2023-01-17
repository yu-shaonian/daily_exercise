import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

class PatchEmbedding(nn.Module):
    def __int__(self,x,patch_size1,patch_size2,dim):
        super().__init__()
        batch, channel, image_width, image_height = x.shape
        assert image_width % patch_size1 == 0 and image_height % patch_size2 == 0 \
        , "Image size must be divided by hte patch size"
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2

        num_patchs = (image_width * image_height) // (patch_size1 * patch_size2)
        patch_dim = channel * patch_size2 * patch_size1

        self.to_patch_embedding = nn.Linear(patch_dim, dim, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patchs + 1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        # (b,C,H,W) ->(B,num_pa,patch*patch*C)
        x = rearrange(x, 'b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=self.patch_size1, p2=self.patch_size2)
        x1 = self.to_patch_embedding(x)
        # (b,C,H,W) ->(B,num_pa,patch*patch*C)

        cls_token = repeat(self.cls_token,'() n d -> b n d', b=b)
        x2 = torch.cat([cls_token,x1], dim=1)
        x2 = x2 + self.pos_embedding

        return x2

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.dim)
        attention_score = nn.Softmax(dim=1)(attention_score)
        out = torch.bmm(attention_score, V)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, project_out=False):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout




