# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -------------------------
# Encoder factory (no pretrained weights)
# -------------------------
def build_encoder(arch: str):
    """
    Returns (model, is_vit).
    - torchvision ResNets: 'resnet18', 'resnet34', 'resnet50', 'resnet101', ...
    - timm ViTs: 'vit_small_patch16_224', 'vit_base_patch16_224', ...
    """
    arch = arch.lower()
    
    # make sure arch in resnet18, resnet50, vit_s, vit_b
    assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101',
                    'vit_s', 'vit_b'], f"Unknown architecture '{arch}'"
    
    if arch == 'vit_s':
        arch = 'vit_small_patch16_224'
    elif arch == 'vit_b':
        arch = 'vit_base_patch16_224'

    # Prefer timm if the name matches a timm model (covers ViT small/base cleanly)
    if arch in timm.list_models():
        # num_classes=0 => remove classifier; forward returns features
        model = timm.create_model(arch, pretrained=False, num_classes=0)
        # Heuristic: ViTs in timm have .patch_embed and .blocks
        is_vit = hasattr(model, "patch_embed") and hasattr(model, "blocks") and ("vit" in arch)
        return model, is_vit

def model_feature_dim(model) -> int:
    """Robustly get the feature dimension after global pooling / CLS."""
    # timm models expose num_features reliably
    if hasattr(model, "num_features"):
        return int(model.num_features)

    # torchvision ResNet after replacing fc with Identity -> in_features is lost,
    # so we infer by a quick forward on a dummy input (safe + cheap).
    # Use a small 224x224 RGB; adjust if your input size differs.
    with torch.no_grad():
        model.eval()
        x = torch.zeros(1, 3, 224, 224)
        y = model(x)
        return int(y.shape[-1])

class MoCo(nn.Module):
    """
    Generic MoCo-style (symmetric) with:
      - a base encoder (query) and a momentum encoder (key)
      - MLP projector on top of encoder outputs
      - MLP predictor on top of projector
    Uses an in-batch contrastive loss (no queue) for simplicity.
    """

    def __init__(
        self,
        encoder_name: str,
        dim: int = 256,
        mlp_dim: int = 4096,
        T: float = 0.2,
        proj_layers: int = 3,     # 2 for ResNet-like, 3 sometimes used for ViT
        pred_layers: int = 2,
    ):
        """
        encoder_name: string name for the base encoder
        dim:         projector output dimension
        mlp_dim:     hidden dimension for MLPs
        T:           softmax temperature
        proj_layers: number of layers in projector MLP
        pred_layers: number of layers in predictor MLP
        """
        super().__init__()
        self.T = T

        # Build encoders (no pretrained weights)
        q_enc, is_vit_q = build_encoder(encoder_name)
        k_enc, is_vit_k = build_encoder(encoder_name)
        assert is_vit_q == is_vit_k, "Query and key encoder types must match"

        self.base_encoder = q_enc
        self.momentum_encoder = k_enc
        self.is_vit = is_vit_q

        feat_dim = model_feature_dim(self.base_encoder)

        # Projectors (on top of encoder outputs)
        self.projector_q = self._build_mlp(
            num_layers=proj_layers, input_dim=feat_dim, mlp_dim=mlp_dim, output_dim=dim,
            last_bn=True
        )
        self.projector_k = self._build_mlp(
            num_layers=proj_layers, input_dim=feat_dim, mlp_dim=mlp_dim, output_dim=dim,
            last_bn=True
        )

        # Predictor
        self.predictor = self._build_mlp(
            num_layers=pred_layers, input_dim=dim, mlp_dim=mlp_dim, output_dim=dim, 
            last_bn=True if self.is_vit else False  # ViT paper uses last BN
        )

        # Initialize momentum encoder weights
        for pb, pm in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            pm.data.copy_(pb.data)
            pm.requires_grad = False

        for pb, pm in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            pm.data.copy_(pb.data)
            pm.requires_grad = False

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        layers = []
        for l in range(num_layers):
            in_d = input_dim if l == 0 else mlp_dim
            out_d = output_dim if l == num_layers - 1 else mlp_dim
            layers.append(nn.Linear(in_d, out_d, bias=False))
            if l < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_d))
                layers.append(nn.ReLU(inplace=True))
            elif last_bn:
                # SimCLR-style last BN without affine
                layers.append(nn.BatchNorm1d(out_d, affine=False))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _update_momentum(self, m: float):
        """Momentum update for encoder and projector_k."""
        for pb, pm in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            pm.data.mul_(m).add_(pb.data, alpha=1.0 - m)
        for pb, pm in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            pm.data.mul_(m).add_(pb.data, alpha=1.0 - m)

    def _encode(self, encoder, projector, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes images to projector space (no predictor here).
        Output shape: [N, dim]
        """
        z = encoder(x)          # features
        if z.ndim > 2:
            z = z.view(z.size(0), -1)
        z = projector(z)
        return z

    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # similarity matrix [N, N]
        logits = torch.einsum("nc,mc->nm", q, k) / self.T

        N = logits.size(0)
        device = logits.device
        labels = torch.arange(N, device=device, dtype=torch.long)

        loss = F.cross_entropy(logits, labels)
        return loss * (2 * self.T)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, m: float):
        """
        x1, x2: two augmented views
        m: momentum coefficient for the EMA update
        """
        # query branch with predictor
        z1_q = self._encode(self.base_encoder, self.projector_q, x1)
        z2_q = self._encode(self.base_encoder, self.projector_q, x2)
        q1 = self.predictor(z1_q)
        q2 = self.predictor(z2_q)

        with torch.no_grad():
            self._update_momentum(m)
            k1 = self._encode(self.momentum_encoder, self.projector_k, x1)
            k2 = self._encode(self.momentum_encoder, self.projector_k, x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    import time
    for i in ["resnet18", "resnet34", "resnet50", "resnet101", "vit_s", "vit_b"]:
        m, is_vit = build_encoder(i)
        print(f"{i}: is_vit={is_vit}, feature_dim={model_feature_dim(m)}")
    
    moco = MoCo(encoder_name="vit_b")
    x = torch.randn(256, 3, 224, 224)
    start_time = time.time()
    loss = moco(x, x, m=0.99)
    end_time = time.time()
    print("Loss:", loss.item())
    print("Forward pass time:", end_time - start_time)