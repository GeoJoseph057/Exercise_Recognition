"""
Pose-based Exercise Recognition Models
Implements LSTM and Vision Transformer architectures for temporal pose sequence classification
"""

import torch
import torch.nn as nn
import math


class PoseLSTM(nn.Module):
    """
    LSTM-based classifier for pose sequences
    Input: (batch, sequence_length, num_keypoints * dimensions)
    """
    def __init__(
        self,
        input_dim=33*3,  # MediaPipe: 33 keypoints * (x, y, visibility)
        hidden_dim=256,
        num_layers=2,
        num_classes=10,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input embedding
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state (or mean pooling)
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        # Classify
        logits = self.classifier(hidden)
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PoseViT(nn.Module):
    """
    Vision Transformer for pose sequence classification
    Treats temporal sequence as tokens similar to image patches
    """
    def __init__(
        self,
        input_dim=33*3,  # MediaPipe: 33 keypoints * (x, y, visibility)
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        num_classes=10,
        dropout=0.1,
        max_seq_len=100
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Patch embedding (frame embedding in our case)
        self.patch_embed = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (better for deep networks)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize cls token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Embed patches (frames)
        x = self.patch_embed(x)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, embed_dim)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len+1, embed_dim)

        # Use class token for classification
        cls_output = x[:, 0]  # (batch, embed_dim)
        cls_output = self.norm(cls_output)

        # Classification
        logits = self.head(cls_output)  # (batch, num_classes)

        return logits


class HybridPoseModel(nn.Module):
    """
    Hybrid model: CNN for spatial features + Transformer for temporal
    Best of both worlds
    """
    def __init__(
        self,
        input_dim=33*3,
        spatial_dim=128,
        embed_dim=256,
        depth=4,
        num_heads=8,
        num_classes=10,
        dropout=0.1
    ):
        super().__init__()

        # Spatial feature extractor (per-frame)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, spatial_dim),
            nn.LayerNorm(spatial_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Temporal transformer
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Extract spatial features per frame
        x = x.view(-1, x.size(-1))  # (batch*seq_len, input_dim)
        x = self.spatial_encoder(x)  # (batch*seq_len, embed_dim)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Temporal modeling
        x = self.transformer(x)  # (batch, seq_len, embed_dim)

        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, embed_dim)

        # Classify
        logits = self.classifier(x)

        return logits


def get_model(model_type, num_classes, config):
    """
    Factory function to create models

    Args:
        model_type: 'lstm', 'vit', or 'hybrid'
        num_classes: number of exercise classes
        config: configuration dict
    """
    if model_type == 'lstm':
        return PoseLSTM(
            input_dim=33*3,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            num_classes=num_classes,
            dropout=config.get('dropout', 0.3),
            bidirectional=config.get('bidirectional', True)
        )
    elif model_type == 'vit':
        return PoseViT(
            input_dim=33*3,
            embed_dim=config.get('embed_dim', 256),
            depth=config.get('depth', 6),
            num_heads=config.get('num_heads', 8),
            num_classes=num_classes,
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'hybrid':
        return HybridPoseModel(
            input_dim=33*3,
            embed_dim=config.get('embed_dim', 256),
            depth=config.get('depth', 4),
            num_heads=config.get('num_heads', 8),
            num_classes=num_classes,
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Test the models
if __name__ == "__main__":
    batch_size = 8
    seq_len = 30
    input_dim = 33 * 3
    num_classes = 5

    # Test data
    x = torch.randn(batch_size, seq_len, input_dim)

    print("Testing PoseLSTM...")
    model_lstm = PoseLSTM(input_dim=input_dim, num_classes=num_classes)
    out_lstm = model_lstm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_lstm.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_lstm.parameters()):,}\n")

    print("Testing PoseViT...")
    model_vit = PoseViT(input_dim=input_dim, num_classes=num_classes)
    out_vit = model_vit(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_vit.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_vit.parameters()):,}\n")

    print("Testing HybridPoseModel...")
    model_hybrid = HybridPoseModel(input_dim=input_dim, num_classes=num_classes)
    out_hybrid = model_hybrid(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_hybrid.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_hybrid.parameters()):,}")