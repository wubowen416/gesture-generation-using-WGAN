import argparse
import datetime
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import math
import joblib as jl
import numpy as np
from modules import AudioGestureSimilarityNet
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter


def bidirectional_margin_ranking_loss(x: torch.Tensor, margin: float):
    """Compute bidirectional_margin_ranking_loss defined in paper https://arxiv.org/pdf/2007.10639.pdf
    Args:
        x: Tensor, shape [B*B, 1]
        margin: float, how far away to put pairs and mismatched pairs
    Note:
        For paris A and B of batch size N,
        the x should be the similarity calculated using A and B,
        such that x = [s(A[0],B[0]), ..., s(A[0],B[N]), s(A[1],B[0]), ..., s(A[1],B[N]), ..., s(A[N],B[1]), ..., s(A[N],B[N])].
    """
    N = int(math.sqrt(x.shape[0]))
    x = x.view(N, N)
    diag = torch.diag(x, diagonal=0)
    # Get off-diagnal elements of x
    off_diag = x.view(-1)[1:].view(N-1, N+1)[:, :-1].contiguous().view(N, N-1)
    mismatchs = off_diag.T.unsqueeze(1).repeat(1, 2, 1).view(-1, N)
    matchs = diag.unsqueeze(0)
    diff = (mismatchs - matchs + margin).unsqueeze(-1)
    loss = torch.max(torch.cat([torch.zeros_like(diff), diff], dim=2), dim=2).values.sum(dim=0).mean()
    return loss


def calculate_rank_n_accuracy(outputs, n=1):
    """Args:
        x: Tensor, shape [B, B, 1]
        n: rank
    Note:
        For paris A and B of batch size N,
        such that x = [[s(A[0],B[0]), ..., s(A[0],B[N])], [s(A[1],B[0]), ..., s(A[1],B[N])], ..., [s(A[N],B[1]), ..., s(A[N],B[N])]].
    """
    y_pred_n = np.argsort(-outputs, axis=1)[:, :n]
    y_true = np.arange(outputs.shape[0])[:, np.newaxis]
    n_false = np.sum(np.sum(((y_true - y_pred_n) == 0), axis=1) == 0)
    acc = 1 - n_false / outputs.shape[0]
    return acc


class RetrievalDataset:

    def __init__(self, data_dir: str, chunk_len: int = 34, stride_len: int = 1, valid_size: float = 0.1) -> None:

        class TorchDataset(Dataset):
            def __init__(self, X, Y):
                super(TorchDataset, self).__init__()
                self.X = X
                self.Y = Y
            def __len__(self):
                return len(self.X)
            def __getitem__(self, i):
                return torch.FloatTensor(self.X[i]), torch.FloatTensor(self.Y[i])

        # Load scaler
        speech_scaler = jl.load(f'{data_dir}/speech_scaler.jl')
        motion_scaler = jl.load(f'{data_dir}/motion_scaler.jl')

        # Load data
        with open(f'{data_dir}/X_train.p', 'rb') as f:
            speechs = pickle.load(f)
        with open(f'{data_dir}/Y_train.p', 'rb') as f:
            motions = pickle.load(f)

        speechs_scaled = list(map(speech_scaler.transform, speechs))
        motions_scaled = list(map(motion_scaler.transform, motions))

        speech_chunks = self.chunkize_batch(speechs_scaled, chunk_len, stride_len)
        motion_chunks = self.chunkize_batch(motions_scaled, chunk_len, stride_len)

        speech_chunks_train, speech_chunks_valid, motion_chunks_train, motion_chunks_valid = train_test_split(
            speech_chunks, motion_chunks, test_size=valid_size, shuffle=True)

        self.train_dataset = TorchDataset(speech_chunks_train, motion_chunks_train)
        self.valid_dataset = TorchDataset(speech_chunks_valid, motion_chunks_valid)

        # Load test data
        with open(f'{data_dir}/X_dev.p', 'rb') as f:
            speechs = pickle.load(f)
        with open(f'{data_dir}/Y_dev.p', 'rb') as f:
            motions = pickle.load(f)

        speechs_scaled = list(map(speech_scaler.transform, speechs))
        motions_scaled = list(map(motion_scaler.transform, motions))

        speech_chunks = self.chunkize_batch(speechs_scaled, chunk_len, stride_len)
        motion_chunks = self.chunkize_batch(motions_scaled, chunk_len, stride_len)

        self.test_dataset = TorchDataset(speech_chunks, motion_chunks)

        self.d_speech = speech_scaler.mean_.shape[0]
        self.d_motion = motion_scaler.mean_.shape[0]
    
    def get_train_dataset(self):
        return self.train_dataset

    def get_valid_dataset(self):
        return self.valid_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_dims(self):
        return self.d_speech, self.d_motion

    def chunkize_batch(self, X: list[np.array], chunk_len: int, stride_len: int):
        return np.concatenate([self.chunkize(x, chunk_len, stride_len) for x in X if x.shape[0] > chunk_len], axis=0)
    
    @staticmethod
    def chunkize(x: np.array, chunk_len: int, stride_len: int):
        num_chunks = (x.shape[0] - chunk_len) // stride_len + 1
        return np.array([x[i_chunk * stride_len:(i_chunk * stride_len) + chunk_len] for i_chunk in range(num_chunks)])

    def make_train_batch(self, speechs, motions):
        B, T, _ = speechs.shape
        speechs = speechs.unsqueeze(1).repeat([1, B, 1, 1]).view(-1, T, self.d_speech)
        motions = motions.unsqueeze(0).repeat([B, 1, 1, 1]).view(-1, T, self.d_motion)
        return speechs, motions


def train(args, log_dir, date):

    data = RetrievalDataset(args.data_dir, args.chunk_len, args.stride_len, valid_size=args.valid_size)
    d_speech, d_motion = data.get_dims()

    train_data_loader = DataLoader(data.get_train_dataset(), batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_data_loader = DataLoader(data.get_valid_dataset(), batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = AudioGestureSimilarityNet(
        d_audio=d_speech,
        d_motion=d_motion,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        activation=args.activation,
        dropout=args.dropout).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.n_epochs//5)

    best_loss = np.inf
    train_writer = SummaryWriter(log_dir=f'{log_dir}/tb_{date}/train')
    valid_writer = SummaryWriter(log_dir=f'{log_dir}/tb_{date}/valid')
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch+1}:")
        # Training
        train_loss = 0
        for speechs, motions in tqdm(train_data_loader):
            speechs, motions = data.make_train_batch(speechs, motions)
            speechs = speechs.to(args.device)
            motions = motions.to(args.device)
            # Step
            model.train()
            model.zero_grad()
            outputs, _, _ = model(speechs, motions)
            loss = bidirectional_margin_ranking_loss(outputs, args.margin)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(train_data_loader)
        scheduler.step()

        # Validating
        valid_loss = 0
        for speechs, motions in valid_data_loader:
            speechs, motions = data.make_train_batch(speechs, motions)
            speechs = speechs.to(args.device)
            motions = motions.to(args.device)
            # Step
            model.eval()
            with torch.no_grad():
                outputs, _, _ = model(speechs, motions)
            loss = bidirectional_margin_ranking_loss(outputs, args.margin)
            valid_loss += loss.item()
        valid_loss /= len(valid_data_loader)
        
        # Save model
        if valid_loss < best_loss :
            best_loss = valid_loss
            torch.save(model.state_dict(), f'{log_dir}/best.pt')

        # Tensorboard log
        train_writer.add_scalar('loss', train_loss, epoch)
        valid_writer.add_scalar('loss', valid_loss, epoch)

        print(f"train_loss: {train_loss:6f}, valid_loss: {valid_loss:6f}")

    
def test(args, log_dir, device='cpu'):
 
    # Load data
    data = RetrievalDataset(args.data_dir, args.chunk_len, args.chunk_len, valid_size=args.valid_size)
    dataset = data.get_test_dataset()[:]
    num_samples = len(dataset[0])
    d_speech, d_motion = data.get_dims()

    # Load model
    model = AudioGestureSimilarityNet(
        d_audio=d_speech,
        d_motion=d_motion,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        activation=args.activation,
        dropout=args.dropout).to(device)

    model.load_state_dict(torch.load(f'{log_dir}/best.pt'))

    # Predicting
    speechs, motions = data.make_train_batch(*dataset)
    model.eval()
    with torch.no_grad():
        speechs = speechs.to(device)
        motions = motions.to(device)
        outputs, _, _ = model(speechs, motions)

    outputs = outputs.view(num_samples, num_samples).numpy()

    rank_1_acc = calculate_rank_n_accuracy(outputs, n=1)
    rank_3_acc = calculate_rank_n_accuracy(outputs, n=3)
    rank_5_acc = calculate_rank_n_accuracy(outputs, n=5)

    print("Rank@1: {:.3f}%".format(rank_1_acc*100))
    print("Rank@3: {:.3f}%".format(rank_3_acc*100))
    print("Rank@5: {:.3f}%".format(rank_5_acc*100))
    print(f"n_smaples: {num_samples}")

    with open(f'{log_dir}/rank@n.txt', 'w') as f:
        f.write(f"Rank 1: {rank_1_acc*100}\n")
        f.write(f"Rank 3: {rank_3_acc*100}\n")
        f.write(f"Rank 5: {rank_5_acc*100}\n")
        f.write(f"n_smaples: {num_samples}\n")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train retrieval net.')

    parser.add_argument('--device', default='cuda', type=str, help="Device for running pytorch model.")
    parser.add_argument('--seed', default=1, type=int, help="Reproducibility.")

    parser.add_argument('--data_dir', default='./data/takekuchi/processed/prosody_hip', type=str, help="Processed data path.")
    parser.add_argument('--chunk_len', default=34, type=int, help="Length for making data chunks.")
    parser.add_argument('--stride_len', default=17, type=int, help="Stride for making data chunks.")

    parser.add_argument('--valid_size', default=0.1, type=float, help="Ratio of data for validation during training.")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size for training.")
    parser.add_argument('--n_epochs', default=100, type=int, help="Total epoch for training.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument('--margin', default=0.05, type=float, help="Margin for training in loss funtion.")

    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--num_encoder_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = 'evaluation/retrieval_net/logs/' + date
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    train(args, log_dir, date)
    test(args, log_dir)