import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class qValuePredictor(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int = 1,
        hidden_dims : list = [16, 16, 32],
    ) -> None:
        super(qValuePredictor, self).__init__()

        # input is of size (batch_size, channels, image_size, image_size)
        # First convolution changes image_size to floor((image_size + 2 * padding - kernel_size) // stride) + 1
        # Build a encoder-decoder architecture with maxpools in between and LeakyReLU as the activation function

        self.step = 0
        self.encoder_store = []

        self.encoder_store.append(
            self.encoder_conv_layer(
                in_channels, 
                hidden_dims[0], 
                5, # Size of kernels
                1, # Number of strides
                3  # Padding given
            )
        )
        
        for i in range(len(hidden_dims) - 1):
            self.encoder_store.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size = 3,
                        padding = 1
                    ),
                    nn.ReLU()
                )
            )
        
        self.linear_layer = (
            nn.Sequential(
                nn.Linear(
                    in_features= hidden_dims[-1] * 25,
                    out_features = 200,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features = 200,
                    out_features = 50,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features = 50,
                    out_features = 4,
                )
            )
        )

        self.encoder = nn.Sequential(*self.encoder_store)

    def encoder_conv_layer(
        self,
        input_channels : int,
        output_channels : int,
        kernel_size : int,
        stride : int = 1,
        padding : int = 0,
        ) -> nn.Sequential:
        return nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size= kernel_size,
                    stride = stride,
                    padding = padding
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size = 2,
                    stride = 2
                ),
        )

    def decoder_conv_layer(
        self,
        input_channels : int,
        output_channels : int,
        kernel_size : int,
        stride : int = 1,
        padding : int = 0,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = "nearest"
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride = stride,
                padding = padding
            ),
        )
    
    # def reconstruction(
    #     self,
    #     input : torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     batch_size, num_channels, w, h = input.shape

    #     x = self.encoder(input)
    #     x = x.view(batch_size, -1)
    #     x = self.encoder_decoder_linear(x)
    #     x = x.view(-1, self.stored_channels, 5, 5)
    #     x = self.decoder2(x)

    #     return x
    
    def forward(
        self,
        input : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, num_channels, w, h = input.shape

        x = self.encoder(input)
        x = x.view(batch_size, -1)
        x = self.linear_layer(x)

        max_q, action_taken = torch.max(x, dim = -1)
        # # max_val is the maximal value 
        # # max_idx is the index of the point (where to travel to but not filtered)
        # max_q, max_idx = torch.max(x.view(x.size(0), -1), dim = -1)
        # # print(max_q)
        # # print(max_idx)

        return max_q, action_taken

    def choose_travel(
        self,
        input : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, w, h = input.shape
        x = self.encoder(input)
        x = x.view(batch_size, -1)
        x = self.linear_layer(x)

        max_q, action_taken = torch.max(x, dim = -1)
        # print(max_q)
        # print(max_idx)
        return max_q, action_taken

    # Find Q value function takes in the image inputs with different channels 
    def find_Q_value(
        self,
        input : torch.Tensor, #These are all the channels sent through the encoder-decoder,
        action : torch.Tensor,
    ):
        batch_size, num_channels, w, h = input.shape
        x = self.encoder(input)
        x = x.view(batch_size, -1)
        x = self.linear_layer(x)

        batch_indices = torch.arange(x.size(0), device='cuda:0').long()
        # print("Q Value found", x)
        # print("Batch indices", batch_indices)
        # print("Actions we are taking", action)
        actions = action.long()
        q_val = x[batch_indices, actions]
        return q_val
    
    def find_loss(
        self,
        reconstructed : torch.Tensor,
        actual : torch.Tensor,
    ) -> torch.Tensor:
        l = F.mse_loss(reconstructed, actual)
        wandb.log({"Loss over time" : l})
        return l
    
    # def reconstruction_loss(
    #     self,
    #     reconstructed : torch.Tensor,
    #     actual : torch.Tensor,
    # ) -> torch.Tensor:
        
    #     l = F.mse_loss(reconstructed, actual)
    #     return l
    
    # def degenerate_reconstruction_loss(
    #     self,
    #     actual : torch.Tensor,
    # ) -> torch.Tensor:
    #     reconstructed = torch.full((actual.shape), 0.0).to(device)
    #     l = F.mse_loss(reconstructed, actual)
    #     return l