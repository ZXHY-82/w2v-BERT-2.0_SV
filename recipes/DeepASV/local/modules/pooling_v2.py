import torch
import torch.nn as nn
import torch.nn.functional as F


class GSP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        ):
        super(GSP, self).__init__()
        self.expansion = 2
    
    def forward(
        self, 
        x: torch.Tensor,
        ):
        out = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return out


class ASP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        ):
        super(ASP, self).__init__()
        self.expansion = 2
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
            )

    def forward(
        self, 
        x: torch.Tensor,
        ):
        w = self.attention(x.transpose(1, 2)).transpose(1, 2)  # w.shape = [B, T, D]
        mu = torch.sum(x * w, dim=1) # mu.shape = [B, D]
        sg = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5)) # sg.shape = [B, D]
        out = torch.cat([mu, sg], dim=1) # out.shape = [B, 2D]
        return out

