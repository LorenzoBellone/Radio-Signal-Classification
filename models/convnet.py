import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, state_dim=32, num_classes=24):
        super(ConvNet, self).__init__()
        # The first layer is made of 1 sequential convolution + ReLU step. The output is then flattened.
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 25, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(25, 25, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # The other layers are fully connected, with the output dimension that is
        # equal to the number of possible actions we can have.
        self.layer2 = nn.Linear(int(state_dim)*int(state_dim)*25, 500)
        
        self.output = nn.Linear(500, num_classes) 
        
    
    def forward(self, x1):
        out = self.layer1(x1)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.layer2(out))
        
        out = self.output(out)

        return out


def net(num_classes: int):
    model = ConvNet(state_dim=32,
                    num_classes=num_classes)
    return model