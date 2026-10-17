"""
FlorNeuroPulse — Cross-Attention Transformer Training & ONNX Export
Privacy-First: Only scalar biomarker vectors are used; no raw media in training.
"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ------------------------------------------------------------------
# 1. Cross-Attention Fusion Model (Audio 15D + Vision 5D -> 32D shared -> 1D score)
# ------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, audio_dim=15, vision_dim=5, embed_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)

        # Multihead cross-attention: Vision = Query, Audio = Key/Value
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Fusion head -> continuous anomaly index [0,1] via sigmoid at inference
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)  # logit output for BCEWithLogitsLoss
        )

    def forward(self, audio_vec, vision_vec):
        # audio_vec: (B, 15)  vision_vec: (B, 5)
        a = self.audio_proj(audio_vec).unsqueeze(1)   # (B, 1, 32)  -> Key/Value source
        v = self.vision_proj(vision_vec).unsqueeze(1)  # (B, 1, 32)  -> Query source

        # Cross-attention: Q=v, K=a, V=a
        attn_out, _ = self.cross_attn(query=v, key=a, value=a)
        attn_out = self.norm1(v + attn_out)
        ff_out = self.norm2(attn_out + self.ff(attn_out))
        # Pool and project
        pooled = ff_out.squeeze(1)  # (B, 32)
        logit = self.fusion_head(pooled)  # (B, 1)
        return logit.squeeze(-1)  # (B,)

# ------------------------------------------------------------------
# 2. Synthetic Dataset (Healthy vs. Pathological Drift)
# ------------------------------------------------------------------
class SyntheticBiomarkerDataset(Dataset):
    """Generates 15D audio + 5D vision vectors with calibrated drift."""
    def __init__(self, n_samples=5000, seed=42):
        super().__init__()
        np.random.seed(seed)
        self.n = n_samples
        # Healthy baseline: low jitter, stable EAR, low asymmetry
        self.healthy_audio = np.random.normal(loc=0.2, scale=0.05, size=(n_samples, 15)).astype(np.float32)
        self.healthy_vision = np.random.normal(loc=0.5, scale=0.08, size=(n_samples, 5)).astype(np.float32)
        # Pathological drift: elevated jitter/shimmer, reduced EAR, increased asymmetry
        self.patho_audio = np.random.normal(loc=0.65, scale=0.12, size=(n_samples, 15)).astype(np.float32)
        self.patho_vision = np.random.normal(loc=0.85, scale=0.15, size=(n_samples, 5)).astype(np.float32)
        # Labels: 0 = healthy, 1 = pathological
        self.labels = np.concatenate([
            np.zeros(n_samples // 2, dtype=np.float32),
            np.ones(n_samples // 2, dtype=np.float32)
        ])
        # Shuffle
        idx = np.random.permutation(n_samples)
        self.healthy_audio = self.healthy_audio[idx[:n_samples//2]]
        self.healthy_vision = self.healthy_vision[idx[:n_samples//2]]
        self.patho_audio = self.patho_audio[idx[n_samples//2:]]
        self.patho_vision = self.patho_vision[idx[n_samples//2:]]
        self.labels = self.labels[idx]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.labels[idx] == 0:
            a = self.healthy_audio[idx % (self.n//2)]
            v = self.healthy_vision[idx % (self.n//2)]
        else:
            a = self.patho_audio[idx % (self.n//2)]
            v = self.patho_vision[idx % (self.n//2)]
        return (
            torch.from_numpy(a),
            torch.from_numpy(v),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# ------------------------------------------------------------------
# 3. Training Loop (AdamW + BCEWithLogitsLoss)
# ------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticBiomarkerDataset(n_samples=8000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    model = CrossAttentionFusion(audio_dim=15, vision_dim=5, embed_dim=32, num_heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for a_batch, v_batch, y_batch in loader:
            a_batch = a_batch.to(device)
            v_batch = v_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(a_batch, v_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    # Save PyTorch weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/florneuropulse_core.pt")
    print("Saved PyTorch weights -> models/florneuropulse_core.pt")
    return model

# ------------------------------------------------------------------
# 4. Dynamic-Batch ONNX Export (with dynamic axes for streaming)
# ------------------------------------------------------------------
def export_onnx(model: CrossAttentionFusion, path="models/florneuropulse_core.onnx"):
    model.eval()
    dummy_audio = torch.randn(1, 15)
    dummy_vision = torch.randn(1, 5)
    torch.onnx.export(
        model,
        (dummy_audio, dummy_vision),
        path,
        input_names=["audio_15d", "vision_5d"],
        output_names=["anomaly_logit"],
        dynamic_axes={
            "audio_15d": {0: "batch_size"},
            "vision_5d": {0: "batch_size"},
            "anomaly_logit": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    print(f"ONNX export complete -> {path}")

if __name__ == "__main__":
    model = train()
    export_onnx(model)
