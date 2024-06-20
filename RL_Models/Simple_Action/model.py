import torch
import torch.nn.functional as F
import torch.nn as nn

# Policy network trained to predict where to travel next
class ActorCNN(nn.Module):
    def __init__(
        self, 
        input_channels : int,
        out_channels : int,
        hidden_dims : list[int] = None
    ) -> None:
        super(ActorCNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        layers = [ self.convLayer(input_channels, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layers.append(self.convLayer(hidden_dims[i], hidden_dims[i + 1]))

        self.model = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1] * 100, out_channels)


    def convLayer(
        self,
        in_channels : int,
        out_channels : int,
    ) -> torch.nn.modules.container.Sequential:
        
        sequence = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size= 3,
                padding = 1
            ), 
            # nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1), # Added maxpool to extract the more important features of the environment
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

        return sequence
    
    def forward(
        self,
        input : torch.Tensor,
    ) -> int:
        x = self.model(input)
        x = torch.flatten(x, start_dim = 1)
        x = self.output_layer(x)
        x = torch.softmax(x, dim = 0) # Use the softmax to choose the next action
        return x
    

#Value network trained to predict the value of the next positions to travel to
class CriticCNN(nn.Module):
    def __init__(
        self, 
        input_channels : int,
        hidden_dims : list[int] = None
    ) -> None:
        super(CriticCNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        layers = [ self.convLayer(input_channels, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layers.append(self.convLayer(hidden_dims[i], hidden_dims[i + 1]))

        self.model = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1] * 100, 1)


    def convLayer(
        self,
        in_channels : int,
        out_channels : int,
    ) -> torch.nn.modules.container.Sequential:
        
        sequence = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size= 3,
                padding = 1
            ), 
            # nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1), # Added maxpool to extract the more important features of the environment
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

        return sequence
    
    def forward(
        self,
        input : torch.Tensor,
    ) -> int:
        x = self.model(input)
        x = torch.flatten(x, start_dim=1)
        x = self.output_layer(x)
        return x