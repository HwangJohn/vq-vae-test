from __future__ import print_function
from random import Random

import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, RandomSampler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import hydra
from omegaconf import DictConfig, OmegaConf

from model.model import Model
from utils import show, get_img

logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ]))
validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ]))

@hydra.main(config_path="config")
def train(cfg: DictConfig):
    writer = SummaryWriter()

    logger.info(f"conf: {OmegaConf.to_yaml(cfg)}")

    mode = cfg.mode
    train_batch_size = cfg.train_batch_size
    valid_batch_size = cfg.valid_batch_size

    num_training_updates = cfg.num_training_updates

    num_hiddens = cfg.num_hiddens
    num_residual_hiddens = cfg.num_residual_hiddens
    num_residual_layers = cfg.num_residual_layers

    embedding_dim = cfg.embedding_dim
    num_embeddings = cfg.num_embeddings

    commitment_cost = cfg.commitment_cost

    decay = cfg.decay
    learning_rate = cfg.learning_rate

    use_norm = cfg.use_norm


    data_variance = np.var(training_data.data / 255.0)

    if mode == 'local':
        training_sample_data = Subset(training_data, np.arange(10))
        training_sampler = RandomSampler(training_sample_data)

        validation_sapmle_data = Subset(validation_data, np.arange(2))
        validation_sampler = RandomSampler(validation_sapmle_data)
        training_loader = DataLoader(training_sample_data,
                                    sampler=training_sampler,
                                    batch_size=train_batch_size, 
                                    shuffle=False,
                                    pin_memory=True)

        validation_loader = DataLoader(validation_sapmle_data,
                                    sampler=validation_sampler,
                                    batch_size=valid_batch_size,
                                    shuffle=False,
                                    pin_memory=True)            
    else:
        training_loader = DataLoader(training_data,
                                    batch_size=train_batch_size, 
                                    shuffle=True,
                                    pin_memory=True)

        validation_loader = DataLoader(validation_data,
                                    batch_size=valid_batch_size,
                                    shuffle=False,
                                    pin_memory=True)            

    print(f"batch_size: {train_batch_size}, num_training_updates: {num_training_updates}")


    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                num_embeddings, embedding_dim, 
                commitment_cost, decay, use_norm).to(device)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.to(device)

    images, _ = next(iter(validation_loader))
    writer.add_graph(model, images.to(device))
    train_res_recon_error = []
    train_res_perplexity = []
    for i in xrange(num_training_updates):
        model.train()
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

            # for valid
            model.eval()
            with torch.no_grad():
                (valid_originals, _) = next(iter(validation_loader))
                valid_originals = valid_originals.to(device)

                vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
                _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
                valid_reconstructions = model._decoder(valid_quantize)

                # for tensorboard logging
                writer.add_image('valid_reconstructions', make_grid(valid_reconstructions+0.5), i)
                writer.add_image('valid_originals', make_grid(valid_originals+0.5), i)                           

                writer.add_scalar('train/loss', loss.item(), i)
                writer.add_scalar('train/vq_loss', vq_loss.item(), i)
                writer.add_scalar('train/train_res_recon_error', recon_error.item(), i)
                writer.add_scalar('train/train_res_perplexity', perplexity.item(), i)

                writer.add_histogram('train/data', data, i)
                writer.add_histogram('train/data_recon', data_recon, i)            
                writer.add_histogram('train/_vq_vae/_embedding', model._vq_vae._embedding.weight, i)            


if __name__ == "__main__":
    train()