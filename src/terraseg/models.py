import torch
from torch import nn


class CustomLandslideCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64,32,2,stride=2), nn.ReLU(), nn.ConvTranspose2d(32,32,2,stride=2), nn.ReLU(), nn.Conv2d(32,1,1))
    def forward(self, x): return self.decoder(self.encoder(x))


class UNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.Conv2d(128,128,3,padding=1),nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv_up1 = nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.conv_up2 = nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.final_conv = nn.Conv2d(32,1,1)
    def forward(self, x):
        d1=self.down1(x); d2=self.down2(self.pool1(d1)); x=self.bottleneck(self.pool2(d2))
        x=self.conv_up1(torch.cat([self.up1(x),d2],dim=1))
        return self.final_conv(self.conv_up2(torch.cat([self.up2(x),d1],dim=1)))


def build_model(name: str, pretrained: bool = True):
    if name == "custom_cnn": return CustomLandslideCNN()
    if name == "unet": return UNetBaseline()
    if name == "resnet34_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to use resnet34_unet") from exc
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)
    raise ValueError(f"Unknown model: {name}")
