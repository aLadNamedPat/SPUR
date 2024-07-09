import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

class qValuePredictor(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int = 1,
        hidden_dims : list = [32, 16, 16],
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
                3, # Number of strides
                3  # Padding given
            )
        )
        
        for i in range(len(hidden_dims) - 1):
            self.encoder_store.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size = 4,
                        padding = 1
                    ),
                    nn.LeakyReLU()
                )
            )

        self.encoder_decoder_linear = (
            nn.Sequential(
                nn.Linear(
                    in_features = hidden_dims[-1] * 4, 
                    out_features = 500
                ),
                nn.LeakyReLU(),
                nn.Linear(
                    in_features = 500,
                    out_features=hidden_dims[-1] * 4
                ),
                nn.LeakyReLU(),
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
                        kernel_size=4,
                        padding = 1,
                    ),
                    nn.LeakyReLU()
                )
            )

        self.decoder_store.append(
            self.decoder_conv_layer(
                hidden_dims[-1],
                out_channels,
                5,
                3,
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
                nn.LeakyReLU(),
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
            nn.LeakyReLU(),
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
        x = x.view(-1, self.stored_channels, 2, 2)
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
        x = x.view(-1, self.stored_channels, 2, 2)
        x = self.decoder(x)
        
        # print("Q_Values:", x)
        positions = torch.argwhere(input[0,2])

        x[:, :, positions[0, 0], positions[0, 1]] = -float("inf")
        # grid = torchvision.utils.make_grid(x)
        # if self.writer is not None and self.step % 10 == 0:
        #     self.writer.add_image("images", grid, self.step)
        #     self.writer.flush()

        if self.step % 50 == 0:
            wandb.log({"Decoder Output" : [wandb.Image(x, caption=f"Noisy Image")]})
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
        x = x.view(-1, self.stored_channels, 2, 2)
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
        
        l = F.smooth_l1_loss(reconstructed, actual)
        
        return l