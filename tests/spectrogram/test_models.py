import torch
from spectrogram.models import build_backbone, SmallCNN, in_channels_for

def test_in_channels_map():
    assert in_channels_for("A0") == 3
    assert in_channels_for("A1") == 3
    assert in_channels_for("A2") == 12

def test_backbone_forward_3ch():
    m = build_backbone(in_ch=3, pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)

def test_backbone_forward_12ch():
    m = build_backbone(in_ch=12, pretrained=False)
    out = m(torch.randn(2, 12, 224, 224))
    assert out.shape == (2, 3)

def test_smallcnn_forward():
    m = SmallCNN(in_ch=3)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)
