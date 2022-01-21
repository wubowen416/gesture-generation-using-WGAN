import os
import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .generator import PoseGenerator
from .discriminator import ConvDiscriminator
from .kde_score import calculate_kde, calculate_kde_batch
from .utils import compute_derivative, compute_velocity
import wandb

import sys
sys.path.append('.')
from tools.takekuchi_dataset_tool.rot_to_pos import rot2pos

import wandb


torch.backends.cudnn.benchmark = True

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
        self.disc_input_dim = output_dim
        self.disc_hidden = hparams.Model.Discriminator.hidden
        self.disc_batchnorm = hparams.Model.Discriminator.batchnorm
        self.disc_layernorm = hparams.Model.Discriminator.layernorm

        self.use_vel = hparams.Train.use_vel
        self.use_acc = hparams.Train.use_acc
        if self.use_vel:
            self.disc_input_dim += output_dim
        if self.use_acc:
            self.disc_input_dim += output_dim

        # Device
        self.device = hparams.device
        self.run_name = hparams.run_name

    
    def build(self, chkpt_path=None):
        self.gen = PoseGenerator(audio_feature_size=self.cond_dim, noise_size=self.noise_dim, dir_size=self.output_dim, n_poses=self.chunklen, hidden_size=self.gen_hidden, num_layers=self.gen_layers, dropout=self.gen_dropout, layernorm=self.gen_layernorm)
        self.disc = ConvDiscriminator(audio_feature_size=self.cond_dim, n_poses=self.chunklen, dir_size=self.disc_input_dim, hidden_size=self.disc_hidden, batchnorm=self.disc_batchnorm, layernorm=self.disc_layernorm, sa=False)
        self.gen.to(self.device)
        self.disc.to(self.device)
        if chkpt_path:
            # Load chkpt
            self.load(chkpt_path)
            self.gen.eval()
        else:
            # Train
            self.gen.train()
            self.disc.train()
            wandb.init(project='gesture_generation', name=self.run_name)
            wandb.define_metric('kde_rot', summary='max')

        print(f"Num of params: G - {self.gen.count_parameters()}, D - {self.disc.count_parameters()}")
        

    def train(self, data, log_dir, hparams):

        # Train relative
        n_epochs = hparams.Train.n_epochs
        batch_size = hparams.Train.batch_size
        n_critic = hparams.Train.n_critic
        lr = hparams.Train.lr
        # gp
        gp_lambda = hparams.Train.gp_lambda
        gp_zero_center = hparams.Train.gp_zero_center
        # cl
        cl_lambda = hparams.Train.cl_lambda

        # Log relative
        n_iteration = 0
        gen_iteration = 0
        log_gap = hparams.Train.log_gap
        hparams.dump(log_dir)

        self.g_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.d_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)

        # train_loader = DataLoader(
        #     data.get_train_dataset(),
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=4)
        
        for epoch in range(n_epochs):

            print(f"Epoch {epoch}/{n_epochs}")

            train_dataset_generator = data.get_train_dataset()
            
            for train_dataset in train_dataset_generator:

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4
                )

                for idx_batch, (seed, cond, target) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=True):

                    # to device
                    seed = seed.to(self.device)
                    cond = cond.to(self.device)
                    target = target.to(self.device)

                    # Train discriminator
                    self.disc.train()
                    self.d_opt.zero_grad()

                    latent = self.sample_noise(cond.shape[0], device=self.device)
                    with torch.no_grad():
                        gen_outputs = self.gen(seed, latent, cond)

                    # Calculate vel
                    if self.use_vel:
                        target_vel = torch.cat([torch.zeros((target.size(0), 1, target.size(-1)), device=self.device), target[:,1:,:] - target[:,:-1,:]], dim=1)
                        gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                        target_acc = torch.cat([torch.zeros((target_vel.size(0), 1, target_vel.size(-1)), device=self.device), target_vel[:,1:,:] - target_vel[:,:-1,:]], dim=1)
                        gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                        target = torch.cat([target, target_vel, target_acc], dim=-1)
                        gen_outputs = torch.cat([gen_outputs, gen_vel, gen_acc], dim=-1)

                    d_loss = - self.disc(target, cond).mean() + self.disc(gen_outputs, cond).mean()
                    w_distance = -d_loss.item()
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
                    d_loss += gp_lambda * gradient_penalty
                    # --------------------------------------------------------------------------------

                    d_loss.backward()
                    self.d_opt.step()

                    # Train generator
                    # if False:
                    if idx_batch % n_critic == 0:
                        self.gen.train()
                        self.g_opt.zero_grad()
                        latent = self.sample_noise(cond.shape[0], device=self.device)
                        gen_outputs = self.gen(seed, latent, cond)

                        if self.use_vel:
                            gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                            gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                            gen_outputs = torch.cat([gen_outputs, gen_vel, gen_acc], dim=-1)

                        # Loss
                        critic = - self.disc(gen_outputs, cond).mean()

                        # --------------------------------------------------------------------------------
                        # continuity loss
                        if self.use_vel:
                            gen_outputs = gen_outputs[:, :, :self.output_dim]
                        pre_pose_error = F.smooth_l1_loss(
                            gen_outputs[:, :self.seedlen], seed[:, :self.seedlen], reduction='none')
                        pre_pose_error = pre_pose_error.sum(dim=1).sum(dim=1) # sum over joint & time step
                        pre_pose_error = pre_pose_error.mean() # mean over batch samples
                        g_loss = critic + cl_lambda * pre_pose_error
                        # --------------------------------------------------------------------------------
                        g_loss.backward()

                        # Operate grads
                        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), hparams.Train.grad_norm_value)
                        torch.nn.utils.clip_grad_value_(self.gen.parameters(), hparams.Train.grad_clip_value)
                        self.g_opt.step()

                        wandb.log({
                            "w_distance": w_distance,
                            "cl_loss": pre_pose_error.item(),
                            "gp_loss": gradient_penalty.item(),
                            "gen_loss": critic.item(),
                        }, step=n_iteration)

                        # print("Estimated w-distance: {:.4f}".format(w_distance))

                        # Log
                        if gen_iteration > 0 and gen_iteration % log_gap == 0:
                        # if True:
                            print("generate samples")
                            # Generate result on dev set
                            output_list, _, motion_list, _ = self.synthesize_batch(data.get_dev_dataset())
                            # Evaluate KDE
                            output = torch.cat(output_list, dim=0).cpu().numpy()
                            motion = torch.cat(motion_list, dim=0).cpu().numpy()
                            output = data.motion_scaler.inverse_transform(output)
                            motion = data.motion_scaler.inverse_transform(motion)
                            kde_mean, _, kde_se = calculate_kde(output, motion)

                            wandb.log({'kde_rot': kde_mean, 'kde_se': kde_se}, step=n_iteration)

                            # Save model
                            self.save(log_dir, n_iteration)

                        gen_iteration += 1

                    n_iteration += 1

    def save(self, log_dir, n_iteration):
        os.makedirs(os.path.join(log_dir, "chkpt"), exist_ok=True)
        save_path = os.path.join(log_dir, f"chkpt/generator_{n_iteration//1000}k.pt")
        torch.save(self.gen.state_dict(), save_path)

    def load(self, chkpt_path):
        self.gen.load_state_dict(torch.load(chkpt_path, map_location=self.device))
    
    def synthesize_batch(self, batch_data):
        output_list, motion_list, output_chunk_list, indexs = [], [], [], []
        for i, (speech, motion) in enumerate(batch_data):
            if len(motion) < self.chunklen:
                continue
            indexs.append(i)
            output, output_chunks = self.synthesize_sequence(speech)
            output_list.append(output)
            output_chunk_list.append(output_chunks)
            motion_list.append(motion)
        return output_list, output_chunk_list, motion_list, indexs
    
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
        return motion, torch.cat(motion_chunks, dim=0)

    def sample_noise(self, batch_size, device, mean=0, std=1):
        return torch.normal(mean=mean, std=std, size=(batch_size, self.noise_dim), device=device)