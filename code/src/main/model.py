# Dependencies
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
print(os.path.abspath(os.path.dirname(__file__) + "/.."))
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset,DataLoader
from pe_encodings.RoPE import apply_2d_rope
from pe_encodings.SPE import build_2d_sincos_pe
from dataclasses import dataclass

@dataclass
class ViTConfig:
    # Image and patch setup
    patch_size: int = 16
    dimensions: tuple = (224, 224)
    
    # ViT architecture
    C: int = 3
    D: int = 384
    n_heads: int = 8
    n_encoder_layers: int = 12
    num_classes: int = 100
    
    # Dropouts & Training params
    drop_path_rate: float = 0.0
    initial_drop: float = 0.0
    drop_attn_rate: float = 0.0
    drop_MLP_rate: float = 0.0
    drop_embed_rate: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):

        self.num_patches = (self.dimensions[0] // self.patch_size) * (self.dimensions[1] // self.patch_size)
        self.d_key = self.D // self.n_heads
        self.prob_step_per_layer = (self.drop_path_rate - self.initial_drop) / self.n_encoder_layers
        self.H, self.W = (int(self.num_patches ** 0.5),) * 2


DEFAULT_CONFIG = ViTConfig()

# ImagePatcher class, generates image patches from the original image

class ImagePatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=config.C,
            out_channels=config.D,
            kernel_size=config.patch_size,
            stride=config.patch_size,
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

prob_step_per_layer = 0.0 # No stochastic depth, but models were trained with them

class MultiHeadedAttention(nn.Module):
    def __init__(self, config, condition):
        super().__init__()
        self.condition = condition
        self.n_heads = config.n_heads
        self.d_head = config.d_key
        self.D = config.D
        self.H, self.W = config.H, config.W
        # Fused QKV projection
        self.qkv = nn.Linear(config.D, 3 * config.D)
        self.proj = nn.Linear(config.D, config.D)
        self.scale = self.d_head ** 0.5
        #Defining dropout
        self.drop_attn_rate = config.drop_attn_rate
        self.dropout = nn.Dropout(self.drop_attn_rate)

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

        if self.condition.upper() == "ROPE":

            q, k = apply_2d_rope(q, k, self.H, self.W, self.d_head)

        # Scaled Dot Product Attention

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop_attn_rate if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1,2).contiguous().view(B, T, self.D)  # merge heads
        return self.proj(out)



#Main Vision Transformer Class
class VisionTransformer(nn.Module):
    def __init__(self, config, condition):
        super().__init__()
        if condition.upper() not in ["APE", "SPE", "RPT", "ABLATED", "ROPE"]:
            raise ValueError("condition must be in the provided list of valid conditions")

        self.config = config
        self.condition = condition
        self.n_encoder_layers = config.n_encoder_layers
        self.device = config.device
        
        self.patcher = ImagePatcher(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.D))
        self.pos_embedding = nn.Embedding(config.num_patches + 1, config.D)
        
        self.dropout_MLP = nn.Dropout(config.drop_MLP_rate)
        self.embed_drop = nn.Dropout(config.drop_embed_rate)
        
        
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(config.D), MultiHeadedAttention(config,condition), DropPath(initial_drop + i * prob_step_per_layer)])
            for i in range(config.n_encoder_layers)
        
        ])
        self.feed_forward = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(config.D),
                nn.Sequential(nn.Linear(config.D, 4*config.D), nn.GELU(), self.dropout_MLP, nn.Linear(4*config.D, config.D)),
                DropPath(initial_drop + i * prob_step_per_layer)
            ]) for i in range(config.n_encoder_layers)
        ])
        self.final_feed_forward = nn.Linear(config.D, config.num_classes)
        
        
        self.pos_indices = torch.arange(config.num_patches + 1, device=config.device)
        if self.condition == "SPE":
            self.register_buffer(
            "sincos_pe",
            build_2d_sincos_pe(self.config.H, self.config.W, config.D, config.device)
            )
        
        def block_forward(self, out, i):
            LN1, MSA, DP_MSA = self.encoder_layers[i]
            LN2, MLP, DP_MLP = self.feed_forward[i]

            out = out + DP_MSA(MSA(LN1(out)))
            out = out + DP_MLP(MLP(LN2(out)))
            return out
        
        self.block_forward = block_forward

    def forward(self, x, RPI = False, magnitude = 1.0):
        patches = self.patcher(x, RPI)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1).to(self.device)

        if self.condition.upper() == "SPE":
            patches = patches + (self.sincos_pe * magnitude)
            x = torch.cat((cls_tokens + (self.cls_pos * magnitude), patches), dim=1)
        elif self.condition.upper() in ["APE", "RPT"]:
            x = torch.cat((cls_tokens, patches), dim=1) + (self.pos_embedding(self.pos_indices))
        else:
            x = torch.cat((cls_tokens, patches), dim=1)

        x = self.embed_drop(x)
        out = x
        #out.requires_grad_(True)

        for i in range(self.n_encoder_layers):

            out = self.block_forward(self, out ,i)

        out = self.final_feed_forward(out[:, 0])
        return out
    
    def predict(self, dataloader, RPI = False, magnitude = 1.0):
        acc_list = []
        for images,labels in dataloader:
            self.eval()
            with torch.inference_mode():

                # Convert tensors to the GPU for faster training
                images,labels = images.to(self.device), labels.to(self.device)

                # Calculate outputs and loss
                out = self(images, RPI, magnitude)
                dev_loss = F.cross_entropy(out,labels, label_smoothing = 0.1)

                # Get the top predictions of the model
                probs = torch.softmax(out,dim = 1)
                top_preds = probs.argmax(dim=1,keepdims=True).view(-1).to(self.device)

                # Calculate the accuracy of the model on data it's never seen
                correct = (top_preds == labels).sum().item()
                accuracy = correct / labels.shape[0]
                acc_list.append(accuracy)

                del images, labels, out, dev_loss
                torch.cuda.empty_cache()

        mean_acc = numpy.array(acc_list).mean().item()
        return mean_acc