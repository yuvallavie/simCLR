import torch.nn as nn
from torchvision.models import resnet50;

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Supervised_RL(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self,device):
        super(Supervised_RL, self).__init__()

        # ------------------------------------------- #
                    # Initialize our encoder
        # ------------------------------------------- #

        self.encoder = resnet50(pretrained=True).to(device);
        self.n_features = self.encoder.fc.in_features;

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity();




