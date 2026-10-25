"""CBMA-TN-VQC: CNN-BiLSTM-MultiHead Attention + quantum-inspired MPS tensor network + VQC, fused regression head."""
import torch, torch.nn as nn
from .qlayer import QuantumBlock

class MPSLayer(nn.Module):
    """Quantum-inspired Matrix Product State layer: each time step feature map phi(x_t)=[cos,sin]
    is contracted through a chain of bond-dimension-D cores (classical tensor-network simulation)."""
    def __init__(self, n_sites, d_in, bond=8, d_out=8):
        super().__init__()
        self.proj = nn.Linear(d_in, 1)
        self.cores = nn.Parameter(torch.randn(n_sites, 2, bond, bond) * 0.1 + torch.eye(bond).expand(n_sites, 2, bond, bond) * 0.5)
        self.left = nn.Parameter(torch.randn(bond) * 0.1 + 1); self.out = nn.Linear(bond, d_out)
    def forward(self, x):                       # x: (B,T,F)
        a = torch.sigmoid(self.proj(x)).squeeze(-1) * torch.pi / 2
        phi = torch.stack([torch.cos(a), torch.sin(a)], -1)  # (B,T,2)
        v = self.left.expand(x.shape[0], -1)
        for t in range(x.shape[1]):
            M = torch.einsum("bp,pij->bij", phi[:, t], self.cores[t])
            v = torch.einsum("bi,bij->bj", v, M); v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        return self.out(v)

class CBMA_TN_VQC(nn.Module):
    def __init__(self, n_feat=5, lookback=24, hidden=32, heads=4, n_qubits=4, n_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(n_feat, hidden, 3, padding=1), nn.ReLU())
        self.bilstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.att = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.tn = MPSLayer(lookback, n_feat)
        self.q = QuantumBlock(hidden, n_qubits, n_layers)
        self.head = nn.Sequential(nn.Linear(hidden + 8 + n_qubits, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.bilstm(h); a, _ = self.att(h, h, h); c = a.mean(1)
        return self.head(torch.cat([c, self.tn(x), self.q(c)], -1)).squeeze(-1)

def electrolyzer_h2_kgph(power_kw, rated_kw=10000.0, hhv_kwh_per_kg=39.4):
    """Toy PEM electrolyser: part-load Faradaic/stack efficiency curve; H2 output in kg/h."""
    import numpy as np
    load = np.clip(power_kw / rated_kw, 0, 1)
    eff = np.where(load < 0.1, 0.0, 0.75 - 0.15 * (load - 0.5) ** 2)
    return load * rated_kw * eff / hhv_kwh_per_kg
