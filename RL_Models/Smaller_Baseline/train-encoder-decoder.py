import torch
import torchvision.utils as vutils
from torchvision import transforms
import os
from PIL import Image
from buffer import ReplayBuffer
from modified_model import qValuePredictor
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

wandb.init(
    # set the wandb project where this run will be logged
    project="Encoder-Decoder-Testing",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "architecture": "Encoder-Decoder",
    "batch_size" : 32,
    }
)

def train(num_epochs, lr = 0.0001):
    buffer = ReplayBuffer(1000000, 32)
    buffer.load_replay_buffer()
    buffer.split_training_testing(0.8)
    actor = qValuePredictor(4, 1, [16, 32, 64, 64]).to(device)
    encoder_and_decoder_params = list(actor.encoder.parameters()) + \
                                list(actor.encoder_decoder_linear.parameters()) + \
                                list(actor.decoder2.parameters())
    optimizer2 = torch.optim.Adam(encoder_and_decoder_params, lr = lr)

    for i in range(num_epochs):
        obs, test_obs = buffer.encoder_decoder_sampling(128)
        reconstructed = actor.reconstruction(obs)

        optimizer2.zero_grad()
        loss = actor.reconstruction_loss(reconstructed.squeeze(1), obs[:,0])
        loss.backward()
        optimizer2.step()

        # Log the first observation and reconstruction
        first_obs = vutils.make_grid(obs[0, 0].unsqueeze(0), normalize=True, scale_each=True)
        first_recon = vutils.make_grid(reconstructed[0].unsqueeze(0), normalize=True, scale_each=True)

        wandb.log({
            "Observation": wandb.Image(first_obs.cpu()),
            "Reconstruction": wandb.Image(first_recon.cpu())
        })

        wandb.log({
            "Reconstruction loss" : loss
        })

        with torch.no_grad():
            #Testing
            reconstructed_testing = actor.reconstruction(test_obs)

            # Log the first observation and reconstruction
            first_obs = vutils.make_grid(test_obs[0, 0].unsqueeze(0), normalize=True, scale_each=True)
            first_recon = vutils.make_grid(reconstructed_testing[0].unsqueeze(0), normalize=True, scale_each=True)

            wandb.log({
                "Test Observation": wandb.Image(first_obs.cpu()),
                "Test Reconstruction" : wandb.Image(first_recon.cpu())
            })
            test_loss = actor.reconstruction_loss(reconstructed_testing.squeeze(1), test_obs[:,0])

            wandb.log({
                "Testing loss" : test_loss
            })

            degenerate_loss = actor.degenerate_reconstruction_loss(test_obs)

            wandb.log({
                "Degenerate loss" : degenerate_loss
            })


train(10000)