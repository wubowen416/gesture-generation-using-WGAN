import os
import numpy as np
import torch
import wandb
import torch.nn.functional as F

from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
from .generator import PoseGenerator, ImprovedPoseGenerator
from .discriminator import ConvDiscriminator
from .nll_score import calculate_nll, calculate_nll_batch
from .utils import compute_derivative, compute_velocity

torch.backends.cudnn.benchmark = True


generator_models = dict(
    PoseGenerator = PoseGenerator,
    ImprovedPoseGenerator = ImprovedPoseGenerator
)

discriminator_models = dict(
    ConvDiscriminator = ConvDiscriminator
)


class ConditionalWGAN:
    def __init__(self, cond_dim, output_dim, hparams):
        self.run_name = hparams.run_name

        # Model relative
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.chunklen = hparams.Data.chunklen
        self.seedlen = hparams.Data.seedlen
        # Generator
        self.gen_model = generator_models[hparams.Model.Generator.name]
        self.gen_hparams = hparams.Model.Generator.to_dict()
        # Discriminator
        self.disc_input_dim = output_dim
        self.disc_model = discriminator_models[hparams.Model.Discriminator.name]
        self.disc_hparams = hparams.Model.Discriminator.to_dict()

        # Training
        self.lr = hparams.Train.lr
        if 'use_vel' in hparams.Train.keys():
            self.use_vel = hparams.Train.use_vel
            if self.use_vel:
                self.disc_input_dim += output_dim
        else:
            self.use_vel = False

        if 'use_acc' in hparams.Train.keys():
            self.use_acc = hparams.Train.use_acc
            if self.use_acc:
                self.disc_input_dim += output_dim
        else:
            self.use_acc = False

        # Grad operators
        if 'grad_norm_value' in hparams.Train.keys():
            self.grad_norm_value = hparams.Train.grad_norm_value
        else:
            self.grad_norm_value = False
        if 'grad_clip_value' in hparams.Train.keys():
            self.grad_clip_value = hparams.Train.grad_clip_value
        else:
            self.grad_clip_value = False

        # Device
        self.device = hparams.device
    
    def build(self, chkpt_path=None):
        self.gen = self.gen_model(self.cond_dim, self.output_dim, **self.gen_hparams)
        self.disc = self.disc_model(self.cond_dim, self.disc_input_dim, **self.disc_hparams)
        self.gen.to(self.device)
        self.disc.to(self.device)
        self.g_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.d_opt = torch.optim.Adam(self.disc.parameters(), lr=self.lr)
        self.gen_iteration = 0
        if chkpt_path:
            # Load chkpt
            self.load(chkpt_path)
            # wandb.init(project='gesture_generation_takekuchi', id=self.run_id, resume='must')
        else:
            self.run_id = wandb.util.generate_id()
            wandb.init(project='gesture_generation_takekuchi', name=self.run_name, id=self.run_id)

        print(f"Num of params: G - {self.gen.count_parameters()}, D - {self.disc.count_parameters()}")

    def fit(self, data, log_dir, hparams):
        # Train relative
        n_epochs = hparams.Train.n_epochs
        batch_size = hparams.Train.batch_size
        n_critic = hparams.Train.n_critic
        # gp
        gp_lambda = hparams.Train.gp_lambda
        gp_zero_center = hparams.Train.gp_zero_center
        # cl
        cl_lambda = hparams.Train.cl_lambda

        # Log relative
        n_iteration = 0
        log_gap = hparams.Train.log_gap
        hparams.dump(log_dir)

        train_loader = DataLoader(
                data.get_train_dataset(),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
        
        for epoch in range(n_epochs):

            print(f"Epoch {epoch}/{n_epochs}")

            for idx_batch, (seed, cond, target) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=True):

                # to device
                seed = seed.to(self.device)
                cond = cond.to(self.device)
                target = target.to(self.device)

                # Train discriminator
                self.disc.train()
                self.d_opt.zero_grad()

                with torch.no_grad():
                    gen_outputs = self.gen(seed, cond)

                # Calculate vel & acc
                if self.use_vel:
                    target_vel = torch.cat([torch.zeros((target.size(0), 1, target.size(-1)), device=self.device), target[:,1:,:] - target[:,:-1,:]], dim=1)
                    gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                    target = torch.cat([target, target_vel], dim=-1)
                    gen_outputs = torch.cat([gen_outputs, gen_vel], dim=-1)
                    if self.use_acc:
                        target_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), target_vel[:,1:,:] - target_vel[:,:-1,:]], dim=1)
                        gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                        target = torch.cat([target, target_acc], dim=-1)
                        gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)
                elif self.use_acc:
                    target_vel = torch.cat([torch.zeros((target.size(0), 1, target.size(-1)), device=self.device), target[:,1:,:] - target[:,:-1,:]], dim=1)
                    target_acc = torch.cat([torch.zeros((target_vel.size(0), 1, target_vel.size(-1)), device=self.device), target_vel[:,1:,:] - target_vel[:,:-1,:]], dim=1)
                    gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                    gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                    target = torch.cat([target, target_acc], dim=-1)
                    gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)

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
                if idx_batch % n_critic == 0:
                    self.gen.train()
                    self.g_opt.zero_grad()
                    gen_outputs = self.gen(seed, cond)

                    # Calculate vel & acc
                    if self.use_vel:
                        gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                        gen_outputs = torch.cat([gen_outputs, gen_vel], dim=-1)
                        if self.use_acc:
                            gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                            gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)
                    elif self.use_acc:
                        gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                        gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                        gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)

                    # Loss
                    critic = - self.disc(gen_outputs, cond).mean()

                    # --------------------------------------------------------------------------------
                    # continuity loss
                    if self.use_vel or self.use_acc:
                        gen_outputs = gen_outputs[:, :, :self.output_dim]
                    pre_pose_error = F.smooth_l1_loss(gen_outputs[:, :self.seedlen], seed[:, :self.seedlen], reduction='none')
                    pre_pose_error = pre_pose_error.sum(dim=1).sum(dim=1) # sum over joint & time step
                    pre_pose_error = pre_pose_error.mean() # mean over batch samples
                    g_loss = critic + cl_lambda * pre_pose_error
                    # --------------------------------------------------------------------------------
                    g_loss.backward()

                    # Operate grads
                    if self.grad_norm_value :
                        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), hparams.Train.grad_norm_value)
                    if self.grad_clip_value:
                        torch.nn.utils.clip_grad_value_(self.gen.parameters(), hparams.Train.grad_clip_value)
                    self.g_opt.step()

                    wandb.log({
                        "w_distance": w_distance,
                        "cl_loss": pre_pose_error.item(),
                        "gp_loss": gradient_penalty.item(),
                        "gen_loss": critic.item(),
                    }, step=self.gen_iteration)

                    # Log
                    if self.gen_iteration % log_gap == 0:

                        if self.gen_iteration > 0:

                            print("generate samples")
                            # Generate result on dev set
                            output_list, _, motion_list, _ = self.synthesize_batch(data.get_dev_dataset())
                            output_list = [data.motion_scaler.inverse_transform(x.cpu().numpy()) for x in output_list]
                            output_vel_list = [compute_derivative(x) for x in output_list]
                            output_acc_list = [compute_derivative(x) for x in output_vel_list]
                            motion_list = [data.motion_scaler.inverse_transform(x.cpu().numpy()) for x in motion_list]
                            motion_vel_list = [compute_derivative(x) for x in motion_list]
                            motion_acc_list = [compute_derivative(x) for x in motion_vel_list]
                            
                            print('Evaluate KDE...')
                            nll_rot = calculate_nll_batch(output_list, motion_list)
                            nll_vel = calculate_nll_batch(output_vel_list, motion_vel_list)
                            nll_acc = calculate_nll_batch(output_acc_list, motion_acc_list)

                            wandb.log({
                                'nll_rot': nll_rot,
                                'nll_vel': nll_vel,
                                'nll_acc': nll_acc,
                            }, step=self.gen_iteration)

                            # Save model
                            self.save(log_dir)
                            
                    self.gen_iteration += 1


    def fit_generator(self, data, log_dir, hparams):
        # Train relative
        n_epochs = hparams.Train.n_epochs
        batch_size = hparams.Train.batch_size
        n_critic = hparams.Train.n_critic
        # gp
        gp_lambda = hparams.Train.gp_lambda
        gp_zero_center = hparams.Train.gp_zero_center
        # cl
        cl_lambda = hparams.Train.cl_lambda

        # Log relative
        log_gap = hparams.Train.log_gap
        hparams.dump(log_dir)
        
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

                    with torch.no_grad():
                        gen_outputs = self.gen(seed, cond)

                    # Calculate vel & acc
                    if self.use_vel:
                        target_vel = torch.cat([torch.zeros((target.size(0), 1, target.size(-1)), device=self.device), target[:,1:,:] - target[:,:-1,:]], dim=1)
                        gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                        target = torch.cat([target, target_vel], dim=-1)
                        gen_outputs = torch.cat([gen_outputs, gen_vel], dim=-1)
                        if self.use_acc:
                            target_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), target_vel[:,1:,:] - target_vel[:,:-1,:]], dim=1)
                            gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                            target = torch.cat([target, target_acc], dim=-1)
                            gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)
                    elif self.use_acc:
                        target_vel = torch.cat([torch.zeros((target.size(0), 1, target.size(-1)), device=self.device), target[:,1:,:] - target[:,:-1,:]], dim=1)
                        target_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), target_vel[:,1:,:] - target_vel[:,:-1,:]], dim=1)
                        gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                        gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                        target = torch.cat([target, target_acc], dim=-1)
                        gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)

                    d_loss = torch.mean(-self.disc(target, cond) + self.disc(gen_outputs, cond))
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
                    if idx_batch % n_critic == 0:
                        self.gen.train()
                        self.g_opt.zero_grad()
                        gen_outputs = self.gen(seed, cond)

                        # Calculate vel & acc
                        if self.use_vel:
                            gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                            gen_outputs = torch.cat([gen_outputs, gen_vel], dim=-1)
                            if self.use_acc:
                                gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                                gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)
                        elif self.use_acc:
                            gen_vel = torch.cat([torch.zeros((gen_outputs.size(0), 1, gen_outputs.size(-1)), device=self.device), gen_outputs[:,1:,:] - gen_outputs[:,:-1,:]], dim=1)
                            gen_acc = torch.cat([torch.zeros((gen_vel.size(0), 1, gen_vel.size(-1)), device=self.device), gen_vel[:,1:,:] - gen_vel[:,:-1,:]], dim=1)
                            gen_outputs = torch.cat([gen_outputs, gen_acc], dim=-1)

                        # Loss
                        critic = torch.mean(-self.disc(gen_outputs, cond))

                        # --------------------------------------------------------------------------------
                        # continuity loss
                        if self.use_vel or self.use_acc:
                            gen_outputs = gen_outputs[:, :, :self.output_dim]
                        pre_pose_error = F.smooth_l1_loss(gen_outputs[:, :self.seedlen], seed[:, :self.seedlen], reduction='none')
                        pre_pose_error = torch.mean(torch.sum(pre_pose_error, dim=[1, 2])) # sum over joint & time step
                        g_loss = critic + cl_lambda * pre_pose_error
                        # --------------------------------------------------------------------------------
                        g_loss.backward()

                        # Operate grads
                        if self.grad_norm_value:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.gen.parameters(), hparams.Train.grad_norm_value)
                        if self.grad_clip_value:
                            torch.nn.utils.clip_grad_value_(self.gen.parameters(), hparams.Train.grad_clip_value)
                        self.g_opt.step()

                        wandb.log({
                            "w_distance": w_distance,
                            "cl_loss": pre_pose_error.item(),
                            "gp_loss": gradient_penalty.item(),
                            "gen_loss": critic.item(),
                            "grad_norm": grad_norm.item()
                        }, step=self.gen_iteration)

                        # Log
                        if self.gen_iteration > 0 and self.gen_iteration % log_gap == 0:
                        # if True:
                            print("generate samples")
                            # Generate result on dev set
                            output_list, _, motion_list, _ = self.synthesize_batch(data.get_dev_dataset())
                            output_list = [data.motion_scaler.inverse_transform(x.cpu().numpy()) for x in output_list]
                            output_vel_list = [compute_derivative(x) for x in output_list]
                            output_acc_list = [compute_derivative(x) for x in output_vel_list]
                            motion_list = [data.motion_scaler.inverse_transform(x.cpu().numpy()) for x in motion_list]
                            motion_vel_list = [compute_derivative(x) for x in motion_list]
                            motion_acc_list = [compute_derivative(x) for x in motion_vel_list]
                            
                            print('Evaluate KDE...')
                            nll_rot = calculate_nll_batch(output_list, motion_list)
                            nll_vel = calculate_nll_batch(output_vel_list, motion_vel_list)
                            nll_acc = calculate_nll_batch(output_acc_list, motion_acc_list)

                            wandb.log({
                                'nll_rot': nll_rot,
                                'nll_vel': nll_vel,
                                'nll_acc': nll_acc,
                            }, step=self.gen_iteration)

                            # Save model
                            self.save(log_dir)
                        self.gen_iteration += 1

    def save(self, log_dir):
        os.makedirs(os.path.join(log_dir, "chkpt"), exist_ok=True)
        save_path = os.path.join(log_dir, f"chkpt/chkpt_{self.gen_iteration//1000}k.pt")
        info = dict()
        info['gen_state_dict'] = self.gen.state_dict()
        info['disc_state_dict'] = self.disc.state_dict()
        info['g_opt_state_dict'] = self.g_opt.state_dict()
        info['d_opt_state_dict'] = self.d_opt.state_dict()
        info['gen_iteration'] = self.gen_iteration
        info['run_id'] = self.run_id
        info['run_name'] = self.run_name
        torch.save(info, save_path)

    def load(self, chkpt_path):
        info = torch.load(chkpt_path, map_location=self.device)
        # self.gen.load_state_dict(info)
        self.gen.load_state_dict(info['gen_state_dict'])
        self.disc.load_state_dict(info['disc_state_dict'])
        self.g_opt.load_state_dict(info['g_opt_state_dict'])
        self.d_opt.load_state_dict(info['d_opt_state_dcit']) # icmi2021 set this to d_opt_state_dcit
        self.gen_iteration = info['gen_iteration']
        self.run_id = info['run_id']
        self.run_name = info['run_name']
    
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
            with torch.no_grad():
                output = self.gen(seed, cond).squeeze(0)
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

    