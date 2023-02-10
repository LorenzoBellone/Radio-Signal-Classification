import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, state_dim=32, num_classes=24, L=1, C=25):
        super(ConvNet, self).__init__()
        # The first layer is made of 1 sequential convolution + ReLU step. The output is then flattened.
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, C, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_layers = nn.ModuleList()
        for i in range(L-1):
            self.conv_layers.append(
                nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )
            )
        
        # The other layers are fully connected, with the output dimension that is
        # equal to the number of possible actions we can have.
        self.output_layer = nn.Linear(int(state_dim)*int(state_dim)*C, num_classes)
        self.L = L
        
    
    def forward(self, x1):
        out = self.layer1(x1)
        for i in range(self.L-1):
            out = self.conv_layers[i](out)
        out = out.reshape(out.size(0), -1)
        out = self.output_layer(out)

        return out


def net(num_classes: int, L=1, C=25):
    model = ConvNet(state_dim=32,
                    num_classes=num_classes,
                    L=L,
                    C=C)
    return model


class ConvNetBN(nn.Module):
    def __init__(self, state_dim=32, num_classes=24, L=1, C=25):
        super(ConvNetBN, self).__init__()
        # The first layer is made of 1 sequential convolution + ReLU step. The output is then flattened.
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, C, kernel_size=3, stride=1),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )

        self.conv_layers = nn.ModuleList()
        for i in range(L-1):
            self.conv_layers.append(
                nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, stride=1),
                nn.BatchNorm2d(C),
                nn.ReLU()
                )
            )
        
        # The other layers are fully connected, with the output dimension that is
        # equal to the number of possible actions we can have.
        self.linear1 = nn.Sequential(
            nn.Linear(int(state_dim-2*L)*int(state_dim-2*L)*C, 500),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(500, num_classes)
        self.L = L
        
    
    def forward(self, x1):
        out = self.layer1(x1)
        for i in range(self.L-1):
            out = self.conv_layers[i](out)
        out = out.reshape(out.size(0), -1)
        out = self.drop(out)
        out = self.linear1(out)
        out = self.output_layer(out)
        
        return out

def netBN(num_classes: int, L=1, C=25):
    model = ConvNetBN(state_dim=32,
                    num_classes=num_classes,
                    L=L,
                    C=C)
    return model