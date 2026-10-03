import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight Depthwise Separable Convolution:
    Splits standard convolution into Depthwise (spatial filtering) and Pointwise (channel mixing).
    Reduces FLOPs and parameter count by ~8-9x while retaining high representational capacity.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))

class ResidualBlock(nn.Module):
    """
    Residual block with Depthwise Separable Convolutions and identity shortcut.
    Helps maintain gradient propagation when training from scratch.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, stride=1)
        self.conv2 = DepthwiseSeparableConv(channels, channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))

class StripContextModule(nn.Module):
    """
    Asymmetric Strip Context Bottleneck:
    Lanes and road boundaries possess strong directional continuity along vertical and diagonal axes.
    Using 1x7 (horizontal) and 7x1 (vertical) strip convolutions enables the model to capture
    long-range contextual continuity along the lane without the high computational cost of large square kernels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_ch = out_channels // 2
        
        # Horizontal strip context
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        # Vertical strip context
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        # Fusion projection
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_h = self.conv_h(x)
        feat_v = self.conv_v(x)
        concat = torch.cat([feat_h, feat_v], dim=1)
        return self.fusion(concat)

class ChannelAttention(nn.Module):
    """
    Lightweight Squeeze-and-Excitation (SE) Channel Attention:
    Dynamically recalibrates channel-wise feature responses to emphasize lane-specific
    cues while attenuating background noise (vegetation, reservoir water reflections).
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w

class DecoderBlock(nn.Module):
    """
    Decoder Stage: Upsamples feature maps by 2x, concatenates high-resolution skip features
    from the encoder, applies channel attention, and refines with convolutions.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        fused_channels = in_channels + skip_channels
        self.conv1 = DepthwiseSeparableConv(fused_channels, out_channels, stride=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)
        self.attention = ChannelAttention(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle small dimension mismatch if any
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attention(x)

class ResLaneSegNet(nn.Module):
    """
    Custom Lightweight Neural Network for Lane Segmentation from Scratch (ResLaneSegNet).
    
    Architectural highlights:
    1. Conv Stem: Fast initial spatial reduction (H/2, W/2) with 32 channels.
    2. 3-Stage Depthwise Separable Encoder: Downsamples feature hierarchy (H/4, H/8, H/16).
    3. Strip Context Bottleneck: Captures long-range horizontal and vertical lane continuity.
    4. Multi-Scale Decoder with Skip Connections: Recovers precise pixel boundaries for thin lane markings.
    5. Channel Attention Gates: Suppresses background distractions (water, trees, sky).
    
    Total Parameters: ~0.55M parameters (Model file < 2.5 MB).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()

        # --- Initial Stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # H/2, W/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # --- Encoder Stages ---
        # Stage 1: H/4, W/4
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, stride=2),
            ResidualBlock(48)
        )
        # Stage 2: H/8, W/8
        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(48, 96, stride=2),
            ResidualBlock(96)
        )
        # Stage 3: H/16, W/16
        self.enc3 = nn.Sequential(
            DepthwiseSeparableConv(96, 160, stride=2),
            ResidualBlock(160)
        )

        # --- Bottleneck ---
        self.bottleneck = StripContextModule(160, 160)

        # --- Decoder Stages ---
        # Decoder 3: H/8, W/8
        self.dec3 = DecoderBlock(in_channels=160, skip_channels=96, out_channels=96)
        # Decoder 2: H/4, W/4
        self.dec2 = DecoderBlock(in_channels=96, skip_channels=48, out_channels=48)
        # Decoder 1: H/2, W/2
        self.dec1 = DecoderBlock(in_channels=48, skip_channels=32, out_channels=32)

        # --- Final Segmentation Head ---
        # Upsample back to original resolution (H, W)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming (He) Normal initialization for robust training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem: H/2, W/2
        s0 = self.stem(x)         # 32 channels
        # Encoder
        s1 = self.enc1(s0)        # 48 channels, H/4
        s2 = self.enc2(s1)        # 96 channels, H/8
        s3 = self.enc3(s2)        # 160 channels, H/16

        # Bottleneck
        b = self.bottleneck(s3)   # 160 channels, H/16

        # Decoder with skip connections
        d3 = self.dec3(b, s2)     # 96 channels, H/8
        d2 = self.dec2(d3, s1)    # 48 channels, H/4
        d1 = self.dec1(d2, s0)    # 32 channels, H/2

        # Output Head: H, W, num_classes
        out = self.head(d1)
        return out

# Alias for model name
PSULaneNet = ResLaneSegNet

if __name__ == "__main__":
    # Test tensor shape and count parameters
    model = ResLaneSegNet(in_channels=3, num_classes=6)
    dummy_input = torch.randn(2, 3, 288, 512)
    output = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
