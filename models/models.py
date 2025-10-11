# Based On MoCo V3 Paper: https://arxiv.org/abs/2104.02057

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import math, torch, torch.nn as nn

# Originally from https://github.com/facebookresearch/moco-v3
@torch.no_grad()
def build_2d_sincos_pos_embed(grid_size, embed_dim, temperature=10000.):
    h, w = grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')  # PyTorch >= 1.10

    assert embed_dim % 4 == 0
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)

    out_w = torch.einsum('m,d->md', grid_w.flatten(), omega)
    out_h = torch.einsum('m,d->md', grid_h.flatten(), omega)

    pe = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)
    pe = pe[None, :, :]  # [1, H*W, C]
    return pe  # no cls token included

@torch.no_grad()
def moco_vit_init(model: nn.Module):
    # 1) fixed 2D sin-cos pos-embed (freeze)
    # timm ViT: model.pos_embed is [1, 1+N, C], model.patch_embed.grid_size is (H, W)
    H, W = model.patch_embed.grid_size
    C = model.pos_embed.shape[-1]
    pe = build_2d_sincos_pos_embed((H, W), C)
    pe_token = torch.zeros(1, 1, C, dtype=torch.float32)
    pe_full = torch.cat([pe_token, pe], dim=1)  # [1, 1+N, C]
    with torch.no_grad():
        model.pos_embed.copy_(pe_full)
    model.pos_embed.requires_grad_(False)

    # 2) CLS token tiny init
    nn.init.normal_(model.cls_token, std=1e-6)

    # 3) qkv + linear init like MoCo v3
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'attn.qkv' in name or name.endswith('qkv'):
                out, in_ = m.weight.shape  # out = 3*embed_dim, in_ = embed_dim
                val = math.sqrt(6. / float(out // 3 + in_))
                nn.init.uniform_(m.weight, -val, val)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # 4) patch embed proj init + optional freeze first "conv"
    proj = getattr(model.patch_embed, 'proj', None)
    if isinstance(proj, nn.Conv2d):
        # fan-aware uniform per MoCo code: sqrt(6 / (3*patch^2 + embed_dim))
        patch = proj.kernel_size[0]
        embed_dim = model.embed_dim
        val = math.sqrt(6. / float(3 * (patch * patch) + embed_dim))
        nn.init.uniform_(proj.weight, -val, val)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)

# -------------------------
# Encoder factory (no pretrained weights)
# -------------------------
def build_encoder(arch: str, moco_style: bool = False):
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
        is_vit = hasattr(model, "patch_embed") and hasattr(model, "blocks") and ("vit" in arch)
        # Heuristic: ViTs in timm have .patch_embed and .blocks
        if is_vit:
            moco_vit_init(model) if moco_style else None
        return model, is_vit

def model_feature_dim(model) -> int:
    """Robustly get the feature dimension after global pooling / CLS."""
    # timm models expose num_features reliably
    if hasattr(model, "num_features"):
        return int(model.num_features)

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
        moco_style: bool = True
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
        q_enc, is_vit_q = build_encoder(encoder_name, moco_style=moco_style)
        k_enc, is_vit_k = build_encoder(encoder_name, moco_style=moco_style)
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

################################### Classifier for Linear Probing #############################
class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=10, requires_grad=False, eval_mode=True, moco_style: bool = False):
        super(Classifier, self).__init__()
        self.backbone = backbone
        
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

        if moco_style:
            self.backbone.pos_embed.requires_grad_(False)
        
        if eval_mode:
            self.backbone.eval()
        
        self.fc = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    import time
    for i in ["resnet18", "vit_s", "vit_b"]:
        m, is_vit = build_encoder(i)
        print(f"{i}: is_vit={is_vit}, feature_dim={model_feature_dim(m)}")
        moco = MoCo(encoder_name=i, moco_style=True)
        x = torch.randn(256, 3, 224, 224)
        start_time = time.time()
        loss = moco(x, x, m=0.99)
        end_time = time.time()
        print("Loss:", loss.item())
        print("Forward pass time:", end_time - start_time)