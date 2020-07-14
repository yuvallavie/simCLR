import torch.nn as nn
import torch;
from resnet_wider import resnet50x1

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self,device):
        super(SimCLR, self).__init__()

        # ------------------------------------------- #
                    # Initialize our encoder
        # ------------------------------------------- #

        self.encoder = resnet50x1().to(device);
        sd = './checkpoints/resnet50-1x.pth'

        sd = torch.load(sd, map_location='cpu')
        self.encoder.load_state_dict(sd['state_dict'])
        self.n_features = self.encoder.fc.in_features;

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity();

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, 64, bias=False),
        )


