"""
Định nghĩa model PyTorch dùng chung cho Training và Inference.

Cả hai model nhận input (N, 3, 64, 64), RGB, giá trị [0, 1] và trả về logits.
(EfficientNetB0 tự upscale + chuẩn hoá ImageNet bên trong.)
"""
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

IMG_SIZE = 64


def _conv_block(cin: int, cout: int, drop: float) -> nn.Sequential:
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(cout),
        nn.MaxPool2d(2),
    ]
    if drop > 0:
        layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)


class CustomCNN(nn.Module):
    """CNN tự xây (4 khối Conv + BN + MaxPool), input 64x64."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32, 0.0),      # 32x32
            _conv_block(32, 64, 0.2),     # 16x16
            _conv_block(64, 128, 0.3),    # 8x8
            _conv_block(128, 256, 0.3),   # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """Residual block của ResNet-18: 2 Conv 3x3 + shortcut (He et al., 2015)."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18(nn.Module):
    """ResNet-18 tự cài đặt, chỉnh cho ảnh nhỏ 64x64 (train từ đầu, không pretrained).

    Stem: Conv 3x3 stride 2 (thay Conv 7x7 + MaxPool của bản ImageNet).
    4 stage x 2 BasicBlock: 64 -> 128 -> 256 -> 512 kênh; feature map 32 -> 16 -> 8 -> 4.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        cfg = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
        self.layers = nn.Sequential(*[
            nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1))
            for cin, cout, stride in cfg])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


class EfficientNetB0Classifier(nn.Module):
    """EfficientNetB0 pretrained (ImageNet), upscale 64 -> 128 bên trong model.

    BatchNorm của backbone luôn được giữ ở chế độ eval để fine-tune ổn định.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, img_size: int = 128):
        super().__init__()
        self.img_size = img_size
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights).features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(1280),
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, k: int = 3):
        """Mở khoá k block cuối của backbone (trừ BatchNorm)."""
        for block in list(self.backbone.children())[-k:]:
            for m in block.modules():
                if not isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters(recurse=False):
                        p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            frozen = not any(p.requires_grad for p in self.backbone.parameters())
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d) or frozen:
                    m.eval()
        return self

    def forward(self, x):
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.head(self.backbone(x))


def build_model(kind: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if kind == "cnn":
        return CustomCNN(num_classes)
    if kind == "resnet18":
        return ResNet18(num_classes)
    if kind == "efficientnetb0":
        return EfficientNetB0Classifier(num_classes, pretrained=pretrained)
    raise ValueError(f"Model không hợp lệ: {kind}")


def save_checkpoint(model: nn.Module, kind: str, class_names: List[str], path) -> None:
    torch.save({"kind": kind, "class_names": class_names, "state_dict": model.state_dict()}, str(path))


def load_classifier(path, device="cpu") -> Tuple[nn.Module, List[str]]:
    """Load checkpoint .pt -> (model ở chế độ eval, class_names)."""
    ckpt = torch.load(str(path), map_location=device)
    model = build_model(ckpt["kind"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["class_names"]
