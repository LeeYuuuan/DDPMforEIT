import torch
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class CLIPEmbedding:
    def __init__():
        super().__init__()


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        
        self.layernorm_2 = nn.LayerNorm(n_embd)
        
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
        
    
    def forward(self, x):
        #(bs, seq_len, dim)
        residue = x
        
        ### SELF ATTENTION ###
        x = self.layernorm_1(x)
        x = self.attention(x) # false, true
        
        x += residue
        
        ###
        
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        
        x = x * torch.sigmoid(1.702 * x) # quick GELU activation function
        x = self.linear_2(x)
        
        x += residue
        return x 
        
        
class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.embedding = CLIPEmbedding()
        
        self.layers = nn.ModuleList(
            [
                CLIPLayer(12, 768) for i in range(12)
            ]
        )
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, voltages):
        
        # state = self.embedding(voltages)
        state = voltages
        
        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)
        
        return output

model = CLIP()
model(torch.rand(32, 768, ))
# print()