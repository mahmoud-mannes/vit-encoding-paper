# Dependencies
import torch
import torch.nn as nn
from torchvision.transforms import V2 as T
from torch.utils.data import Dataset,DataLoader
from RoPE import apply_2d_rope
from SPE import build_2d_sincos_pe

# ImagePatcher class, generates image patches from the original image

class ImagePatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=C,
            out_channels=D,
            kernel_size=patch_size,
            stride=patch_size,
            padding_mode="zeros"
        )

    def forward(self, x, RPI):
        out = self.conv(x)  # [B, D, H', W']
        out = out.flatten(2).transpose(1, 2)  # [B, L, D]

        if RPI:
            perm = torch.randperm(out.shape[1])
            return out[:,perm]
        return out

class DropPath(nn.Module):
  def __init__(self,d_prob):
    super().__init__()
    self.drop_path_rate = d_prob
  def forward(self,x):
    if not self.training or self.drop_path_rate == 0.0:
      return x
    keep_path_rate = 1 - self.drop_path_rate
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.rand(shape, device = x.device) < keep_path_rate
    x = x * mask / keep_path_rate
    return x
initial_drop = 0.0

prob_step_per_layer = (drop_path_rate - initial_drop) / n_encoder_layers

class MultiHeadedAttention(nn.Module):
    def __init__(self, condition = "None"):
        super().__init__()
        self.condition = condition
        self.n_heads = n_heads
        self.d_head = d_key
        self.D = D
        # Fused QKV projection
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.scale = self.d_head ** 0.5
        #Defining dropout
        self.dropout = nn.Dropout(drop_attn_rate)

    def forward(self, x):
        B, T, _ = x.shape
        # x: [B, T, D]
        qkv = self.qkv(x)  # [B, T, 3*D]
        # Split into Q, K, V and reshape for multi-head: [B, T, n_heads, d_head]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1,2)  # [B, n_heads, T, d_head]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        # Apply RoPE to Q and K

        q, k = apply_2d_rope(q, k, self.H, self.W, self.d_head)

        # Scaled Dot Product Attention

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=drop_attn_rate if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1,2).contiguous().view(B, T, self.D)  # merge heads
        return self.proj(out)



#Main Vision Transformer Class
class VisionTransformer(nn.Module):
    def __init__(self,condition):
        super().__init__()
        if condition.upper() not in ["APE", "SPE", "RPT", "ABLATED", "ROPE"]:
            raise ValueError("condition must be in the provided list of valid conditions")
        self.condition = condition
        
        self.patcher = ImagePatcher()
        self.cls_token = nn.Parameter(torch.randn(1, 1, D))
        self.pos_embedding = nn.Embedding(num_patches + 1, D)
        
        self.dropout_MLP = nn.Dropout(drop_MLP_rate)
        self.embed_drop = nn.Dropout(drop_embed_rate)
        
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(D), MultiHeadedAttention(condition), DropPath(initial_drop + i * prob_step_per_layer)])
            for i in range(n_encoder_layers)
        
        ])
        self.feed_forward = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(D),
                nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), self.dropout_MLP, nn.Linear(4*D, D)),
                DropPath(initial_drop + i * prob_step_per_layer)
            ]) for i in range(n_encoder_layers)
        ])
        self.final_feed_forward = nn.Linear(D, num_classes)
        
        
        self.pos_indices = torch.arange(num_patches + 1, device=device)
        self.register_buffer(
        "sincos_pe",
        build_2d_sincos_pe(self.H, self.W, D, device)
        )
        
        def block_forward(self, out, i):
            LN1, MSA, DP_MSA = self.encoder_layers[i]
            LN2, MLP, DP_MLP = self.feed_forward[i]

            out = out + DP_MSA(MSA(LN1(out)))
            out = out + DP_MLP(MLP(LN2(out)))
            return out
        
        self.block_forward = block_forward

    def forward(self, x, RPI = False):
        patches = self.patcher(x, RPI)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1).to(device)

        if condition == "SPE":
            patches = patches + (self.sincos_pe * magnitude)
            x = torch.cat((cls_tokens + (self.cls_pos * magnitude), patches), dim=1)
        elif condition in ["APE", "RPT"]:
            x = torch.cat((cls_tokens, patches), dim=1) + (self.pos_embedding(self.pos_indices))
        else:
            x = torch.cat((cls_tokens, patches), dim=1)

        x = self.embed_drop(x)
        out = x
        #out.requires_grad_(True)

        for i in range(n_encoder_layers):

            out = self.block_forward(self, out ,i)

        out = self.final_feed_forward(out[:, 0])
        return out
    
    def predict(self, dataset, corruption_type=None):
        pass

