import os
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .generator import PoseGenerator
from .discriminator import ConvDiscriminator
from .kde_score import calculate_kde

class ConditionalWGAN:

    def __init__(self, cond_dim, output_dim, hparams):

        # Model relative
        self.cond_dim = cond_dim
        self.noise_dim = hparams.Model.Generator.noise_dim
        self.output_dim = output_dim
        self.chunklen = hparams.Data.chunklen
        self.seedlen = hparams.Data.seedlen
        # Generator
        self.gen_hidden = hparams.Model.Generator.hidden
        self.gen_layers = hparams.Model.Generator.num_layers
        self.gen_layernorm = hparams.Model.Generator.layernorm
        self.gen_dropout = hparams.Model.Generator.dropout
        # Discriminator
        self.disc_hidden = hparams.Model.Discriminator.hidden
        self.disc_batchnorm = hparams.Model.Discriminator.batchnorm
        self.disc_layernorm = hparams.Model.Discriminator.layernorm

        # Device
        self.device = hparams.device

    
    def build(self):
        self.gen = PoseGenerator(audio_feature_size=self.cond_dim, noise_size=self.noise_dim, dir_size=self.output_dim, n_poses=self.chunklen, hidden_size=self.gen_hidden, num_layers=self.gen_layers, dropout=self.gen_dropout, layernorm=self.gen_layernorm)
        self.disc = ConvDiscriminator(audio_feature_size=self.cond_dim, n_poses=self.chunklen, dir_size=self.output_dim, hidden_size=self.disc_hidden, batchnorm=self.disc_batchnorm, layernorm=self.disc_layernorm, sa=False)
        self.gen.to(self.device)
        self.disc.to(self.device)
        self.g_opt = torch.optim.Adam(self.gen.parameters(), lr=1e-4)
        self.d_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4)
        print(f"Num of params: G - {self.gen.count_parameters()}, D - {self.disc.count_parameters()}")
        

    def train(self, data, log_dir, hparams):

        # Train relative
        n_epochs = hparams.Train.n_epochs
        batch_size = hparams.Train.batch_size
        n_critic = hparams.Train.n_critic
        # gp
        gp_lambda = hparams.Train.gp_lambda
        gp_zero_center = hparams.Train.gp_zero_center
        # cl
        cl_lambda = hparams.Train.cl_lambda

        # Get dataloader
        train_loader = DataLoader(
            data.get_train_dataset(),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)

        # Log relative
        writer = SummaryWriter(log_dir)
        gen_iteration = 0
        log_gap = hparams.Train.log_gap
        
        for epoch in range(n_epochs):

            for idx_batch, (seed, cond, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

                # to device
                seed = seed.to(self.device)
                cond = cond.to(self.device)
                target = target.to(self.device)

                # Train discriminator
                self.disc.train()
                self.gen.eval()
                self.d_opt.zero_grad()
                
                latent = self.sample_noise(batch_size, device=self.device)
                with torch.no_grad():
                    gen_outputs = self.gen(seed, latent, cond)
                n_w_dis = - self.disc(target, cond).mean() + self.disc(gen_outputs, cond).mean()

                # --------------------------------------------------------------------------------
                # Compute gradient penalty
                # Random weight term for interpolation between real and fake samples
                alpha = torch.Tensor(np.random.random((1, 1))).to(self.device)
                # Get random interpolation between real and fake samples
                interpolate = (alpha * target + ((1 - alpha) * gen_outputs)).requires_grad_(True)
                d_interpolate = self.disc(interpolate, cond)
                gradients = autograd.grad(
                    outputs=d_interpolate,
                    inputs=interpolate,
                    grad_outputs=torch.ones_like(d_interpolate),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                gradients = gradients.reshape(gradients.size(0), -1)
                if gp_zero_center:
                    # Zero-centered gradient penalty
                    # Improving Generalization and stability of GAN, Thanh-Tung+ 2019, ICLR
                    gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
                else:
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                d_loss = n_w_dis + gp_lambda * gradient_penalty
                # --------------------------------------------------------------------------------

                d_loss.backward()
                self.d_opt.step()

                # Train generator
                if idx_batch % n_critic == 0:
                    self.disc.eval()
                    self.gen.train()
                    self.g_opt.zero_grad()

                    latent = self.sample_noise(batch_size, device=self.device)
                    gen_outputs = self.gen(seed, latent, cond)

                    # Loss
                    critic = - self.disc(gen_outputs, cond).mean()

                    # --------------------------------------------------------------------------------
                    # continuity loss
                    gen_pre_poses = gen_outputs[:, :self.chunklen]
                    pre_poses = seed[:, :self.chunklen]
                    pre_pose_error = F.smooth_l1_loss(gen_pre_poses, pre_poses, reduction='none')
                    pre_pose_error = pre_pose_error.sum(dim=1).sum(dim=1) # sum over joint & time step
                    pre_pose_error = pre_pose_error.mean() # mean over batch samples
                    g_loss = critic + cl_lambda * pre_pose_error
                    # --------------------------------------------------------------------------------

                    g_loss.backward()
                    self.g_opt.step()
                    gen_iteration += 1

                    w_distance = - n_w_dis.item()
                    # Estimated w-distance (opposite to disc loss)
                    writer.add_scalar("loss/w-distance", w_distance, gen_iteration)
                    # Pre pose error
                    writer.add_scalar("loss/pre-pose-error", pre_pose_error.item(), gen_iteration)
                    # generator loss (critic)
                    writer.add_scalar("loss/gen-loss", critic.item(), gen_iteration)
                    # gradient penalty
                    writer.add_scalar("loss/gradient-penalty", gradient_penalty.item(), gen_iteration)

                # Log
                if gen_iteration > 0 and gen_iteration % log_gap == 0:
                    print("generate samples")
                    # Generate result on dev set
                    output_list, motion_list = [], []
                    for i, (speech, motion) in enumerate(data.get_dev_dataset()):
                        if len(motion) < self.chunklen:
                            continue
                        output = self.synthesize_sequence(speech).cpu().numpy()
                        data.save_unity_result(output, os.path.join(log_dir, f"{gen_iteration//1000}k/motion_{i}.txt"))
                        output_list.append(output)
                        motion_list.append(motion.cpu().numpy())
                    # Evaluate KDE
                    output = np.concatenate(output_list, axis=0)
                    motion = np.concatenate(motion_list, axis=0)
                    kde_mean, _, kde_se = calculate_kde(output, motion)
                    writer.add_scalar("kde/mean", kde_mean, gen_iteration)
                    writer.add_scalar("kde/se", kde_se, gen_iteration)

    
    def synthesize_sequence(self, speech_chunks):
        '''Take speech as input (N, chunklen, dim) as numpy array, assuming scaled and chunkized'''
        self.gen.eval()
        # Generate iteratively
        motion_chunks = []
        seed = torch.zeros(size=(1, self.chunklen, self.output_dim)).to(self.device)
        for cond in speech_chunks.to(self.device):
            cond = cond.unsqueeze(0)
            latent = self.sample_noise(1, device=self.device)
            with torch.no_grad():
                output = self.gen(seed, latent, cond).squeeze(0)
            seed[:, :self.seedlen] = output[-self.seedlen:]
            motion_chunks.append(output)

        # Smooth transition
        motion = motion_chunks[0][:self.chunklen-self.seedlen]
        for i in range(1, len(motion_chunks)):
            trans_prev = motion_chunks[i-1][-self.seedlen:]
            trans_next = motion_chunks[i][:self.seedlen]
            trans_motion = []
            for k in range(self.seedlen):
                trans = ((self.seedlen - k) / (self.seedlen + 1)) * trans_prev[k] + ((k + 1) / (self.seedlen + 1)) * trans_next[k]
                trans_motion.append(trans)
            trans_motion = torch.stack(trans_motion)
            # Append each
            if i != len(motion_chunks) - 1: # not last chunk
                motion = torch.cat([motion, trans_motion, motion_chunks[i][self.seedlen:self.chunklen-self.seedlen]], dim=0)
            else: # last chunk
                motion = torch.cat([motion, trans_motion, motion_chunks[i][self.seedlen:self.chunklen]], dim=0)
        return motion


    def sample_noise(self, batch_size, device, mean=0, std=1):
        return torch.normal(mean=mean, std=std, size=(batch_size, self.noise_dim), device=device)