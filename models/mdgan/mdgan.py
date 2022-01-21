import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .generator import PoseGenerator
from .discriminator import ConvDiscriminator, ConditionalConvDiscriminator
from .kde_score import calculate_kde, calculate_kde_batch
from .utils import compute_derivative, compute_velocity
import wandb

torch.backends.cudnn.benchmark = True

class MultiDConditionalWGAN:

    def __init__(self, d_cond, d_data, hparams):

        # Model relative
        self.d_cond = d_cond
        self.d_data = d_data
        self.chunk_len = hparams.Data.chunklen
        self.seed_len = hparams.Data.seedlen

        # Generator
        self.gen_config = hparams.Model.Generator
        # Conditional Discriminator
        self.cond_disc_config = hparams.Model.CondDiscriminator
        # Pose Discriminator
        self.pose_disc_config = hparams.Model.PoseDiscriminator

        # Device
        self.device = hparams.device

        # Log
        self.run_name = hparams.run_name
    
    def build(self, chkpt_path=None):

        self.gen = PoseGenerator(
            d_cond=self.d_cond,
            d_data=self.d_data,
            **self.gen_config.to_dict()
        )

        self.cond_disc = ConditionalConvDiscriminator(
            d_cond=self.d_cond,
            d_data=self.d_data,
            **self.cond_disc_config.to_dict()
        )

        self.pose_disc = ConvDiscriminator(
            d_data=self.d_data,
            **self.pose_disc_config.to_dict()
        )

        self.gen.to(self.device)
        self.cond_disc.to(self.device)
        self.pose_disc.to(self.device)

        if chkpt_path:
            # Load chkpt
            self.load(chkpt_path)
            self.gen.eval()
        else:
            wandb.init(project='gesture_generation', name=self.run_name)
            wandb.define_metric('kde_rot', summary='max')

        print(f"Num of params: gen - {self.gen.count_parameters()}, cond_disc - {self.cond_disc.count_parameters()}, pose_disc - {self.pose_disc.count_parameters()}")

    def train(self, data, log_dir, hparams):

        # Train relative
        n_epochs = hparams.Train.n_epochs
        batch_size = hparams.Train.batch_size
        n_critic = hparams.Train.n_critic
        lr = hparams.Train.lr
        # cl
        cl_lambda = hparams.Train.cl_lambda

        # Log relative
        n_iteration = 0
        gen_iteration = 0
        log_gap = hparams.Train.log_gap
        hparams.dump(log_dir)

        self.g_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.cond_d_opt = torch.optim.Adam(self.cond_disc.parameters(), lr=lr)
        self.pose_d_opt = torch.optim.Adam(self.pose_disc.parameters(), lr=lr)
        
        for epoch in range(n_epochs):

            print(f"Epoch {epoch + 1}/{n_epochs}")

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

                    # Fake poses
                    self.gen.eval()
                    gen_outputs = self.gen(cond, seed).detach()

                    # Train cond disc
                    self.cond_disc.train()
                    cond_nwd = - self.cond_disc(target, cond).mean() + self.cond_disc(gen_outputs, cond).mean()
                    cond_disc_loss = cond_nwd

                    # Update 
                    self.cond_disc.zero_grad()
                    cond_disc_loss.backward()
                    self.cond_d_opt.step()

                    # Train pose disc
                    self.pose_disc.train()
                    pose_nwd = - self.pose_disc(target).mean() + self.pose_disc(gen_outputs).mean()
                    pose_disc_loss = pose_nwd

                    # Update
                    self.pose_disc.zero_grad()
                    pose_disc_loss.backward()
                    self.pose_d_opt.step()

                    # Train generator
                    self.gen.train()
                    self.cond_disc.eval()
                    self.pose_disc.eval()
                    if idx_batch % n_critic == 0:
                        gen_outputs = self.gen(cond, seed)

                        # Critic
                        cond_critic = - self.cond_disc(gen_outputs, cond).mean()
                        pose_critic = - self.pose_disc(gen_outputs).mean()

                        # --------------------------------------------------------------------------------
                        # continuity loss
                        pre_pose_error = F.smooth_l1_loss(
                            gen_outputs[:, :self.seed_len], seed[:, :self.seed_len], reduction='none')
                        pre_pose_error = pre_pose_error.sum(dim=1).sum(dim=1) # sum over joint & time step
                        pre_pose_error = pre_pose_error.mean() # mean over batch samples
                        # --------------------------------------------------------------------------------

                        # Loss
                        g_loss = (1 * cond_critic + 0.0 * pose_critic) + cl_lambda * pre_pose_error

                        # Update
                        self.gen.zero_grad()
                        g_loss.backward()
                        # Operate grads
                        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), hparams.Train.grad_norm_value)
                        torch.nn.utils.clip_grad_value_(self.gen.parameters(), hparams.Train.grad_clip_value)
                        self.g_opt.step()

                        wandb.log({
                            "cond_w_distance": -cond_nwd.item(),
                            "pose_w_distance": -pose_nwd.item(),
                            "cl_loss": pre_pose_error.item(),
                            "gen_loss": (cond_critic + pose_critic).item(),
                        }, step=n_iteration)

                        # Log
                        if gen_iteration > 0 and gen_iteration % log_gap == 0:
                        # if True:
                            print("generate samples")
                            # Generate result on dev set
                            output_list, _, motion_list, _ = self.synthesize_batch(data.get_dev_dataset())
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
            if len(motion) < self.chunk_len:
                continue
            indexs.append(i)
            output, output_chunks = self.synthesize_sequence(speech)
            output_list.append(output)
            output_chunk_list.append(output_chunks)
            motion_list.append(motion)
        return output_list, output_chunk_list, motion_list, indexs
    
    def synthesize_sequence(self, speech_chunks):
        '''Take speech as input (N, chunk_len, dim) as numpy array, assuming scaled and chunkized'''
        self.gen.eval()
        # Generate iteratively
        motion_chunks = []
        seed = torch.zeros(size=(1, self.chunk_len, self.d_data)).to(self.device)
        for cond in speech_chunks.to(self.device):
            cond = cond.unsqueeze(0)
            with torch.no_grad():
                output = self.gen(cond, seed).squeeze(0)
            seed[:, :self.seed_len] = output[-self.seed_len:]
            motion_chunks.append(output)

        # Smooth transition
        motion = motion_chunks[0][:self.chunk_len-self.seed_len]
        for i in range(1, len(motion_chunks)):
            trans_prev = motion_chunks[i-1][-self.seed_len:]
            trans_next = motion_chunks[i][:self.seed_len]
            trans_motion = []
            for k in range(self.seed_len):
                trans = ((self.seed_len - k) / (self.seed_len + 1)) * trans_prev[k] + ((k + 1) / (self.seed_len + 1)) * trans_next[k]
                trans_motion.append(trans)
            trans_motion = torch.stack(trans_motion)
            # Append each
            if i != len(motion_chunks) - 1: # not last chunk
                motion = torch.cat([motion, trans_motion, motion_chunks[i][self.seed_len:self.chunk_len-self.seed_len]], dim=0)
            else: # last chunk
                motion = torch.cat([motion, trans_motion, motion_chunks[i][self.seed_len:self.chunk_len]], dim=0)
        return motion, torch.cat(motion_chunks, dim=0)