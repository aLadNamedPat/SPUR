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

        self.encoder_decoder_linear = (
            nn.Sequential(
                nn.Linear(
                    in_features = hidden_dims[-1] * 25, 
                    out_features = 500,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features = 500,
                    out_features=hidden_dims[-1] * 25
                ),
                nn.ReLU(),
            )
        )

        self.stored_channels = hidden_dims[-1]

        hidden_dims.reverse()
        self.decoder_store = []

        for i in range(len(hidden_dims) - 1):
            self.decoder_store.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        padding = 1,
                    ),
                    nn.ReLU(),
                )
            )

        self.decoder_store.append(
            self.decoder_conv_layer(
                hidden_dims[-1],
                out_channels,
                5,
                1,
                3
            )
        )

        self.encoder = nn.Sequential(*self.encoder_store)
        self.decoder = nn.Sequential(*self.decoder_store)

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
        
    def forward(
        self,
        input : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, num_channels, w, h = input.shape

        x = self.encoder(input)
        x = x.view(batch_size, -1)
        x = self.encoder_decoder_linear(x)
        x = x.view(-1, self.stored_channels, 5, 5)
        x = self.decoder(x)

        positions = torch.argwhere(input)
        # max_val is the maximal value 
        # max_idx is the index of the point (where to travel to but not filtered)
        max_q, max_idx = torch.max(x.view(x.size(0), -1), dim = -1)
        # print(max_q)
        # print(max_idx)

        return max_q, max_idx

    def choose_travel(
        self,
        input : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, w, h = input.shape
        x = self.encoder(input)
        x = x.view(batch_size, -1)
        x = self.encoder_decoder_linear(x)
        x = x.view(-1, self.stored_channels, 5, 5)
        x = self.decoder(x)
        
        # print("Q_Values:", x) 
        positions = torch.argwhere(input[0,1])

        x[:, :, positions[0, 0], positions[0, 1]] = -float("inf")

        if self.step % 500 == 0:
            input = input.squeeze()
            a = x.flatten()
            x_normalized = ((x - torch.kthvalue(a, 2)[0])/ (x.max() - torch.kthvalue(a, 2)[0]) * 255).squeeze()
            x_normalized[torch.argmin(x_normalized) // 8, torch.argmin(x_normalized) % 8] = 0

            wandb.log({"Decoder Output" : [wandb.Image(x_normalized.squeeze(), caption=f"Decoded")]})
            # wandb.log({"Poisson Distribution" : [wandb.Image(input[0], caption=f"Probability no event occurred there")]})

            # wandb.log({"Agent Location" : [wandb.Image(input[1], caption=f"Location")]})
            wandb.log({"Expected Reward" : [wandb.Image(input[0], caption=f"Tracked expectation")]})

            # wandb.log({"Probability Grid" : [wandb.Image(input[1], caption=f"Tracked probability")]})
            wandb.log({"Agent Position" : [wandb.Image(input[1], caption=f"Agent Position")]})
            # wandb.log({"Probability Map" : [wandb.Image(input)]})
        self.step += 1

        # max_val is the maximal value 
        # max_idx is the index of the point (where to travel to but not filtered)
        max_q, max_idx = torch.max(x.view(x.size(0), -1), dim = -1)
        # print(max_q)
        # print(max_idx)
        return max_q, max_idx

    # Find Q value function takes in the image inputs with different channels 
    def find_Q_value(
        self,
        input : torch.Tensor, #These are all the channels sent through the encoder-decoder,
        action : tuple[int, int],
    ):
        batch_size, num_channels, w, h = input.shape
        x = self.encoder(input)
        x = x.view(batch_size, -1)
        x = self.encoder_decoder_linear(x)
        x = x.view(-1, self.stored_channels, 5, 5)
        x = self.decoder(x)
        action_x = action[0].long()
        action_y = action[1].long()

        batch_indices = torch.arange(x.size(0), device='cuda:0').long()
        q_val = x[batch_indices, 0, action_x, action_y]
        return q_val
    
    def find_loss(
        self,
        reconstructed : torch.Tensor,
        actual : torch.Tensor,
    ) -> torch.Tensor:
        l = F.mse_loss(reconstructed, actual)
        wandb.log({"Loss over time" : l})
        return l
    
    def reconstruction_loss(
        self,
        reconstructed : torch.Tensor,
        actual : torch.Tensor,
    ) -> torch.Tensor:
        
        l = F.mse_loss(reconstructed, actual)
        return l
    
    def degenerate_reconstruction_loss(
        self,
        actual : torch.Tensor,
    ) -> torch.Tensor:
        reconstructed = torch.full((actual.shape), 0.0).to(device)
        l = F.mse_loss(reconstructed, actual)
        return l