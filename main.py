import argparse
import os
import random
import pickle
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

# ========== Utility Functions ==========
def get_experiment_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def log_metrics(metrics, path):
    import csv
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

# ========== Data Utilities ==========
def load_and_split(data_dir='ml-latest-small'):
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user2idx = {u: i+1 for i, u in enumerate(user_ids)}
    movie2idx = {m: i+1 for i, m in enumerate(movie_ids)}
    idx2movie = {i: m for m, i in movie2idx.items()}
    ratings['uid'] = ratings['userId'].map(user2idx)
    ratings['mid'] = ratings['movieId'].map(movie2idx)
    user_seqs = ratings.sort_values(['uid','timestamp']).groupby('uid')['mid'].apply(list).to_dict()
    train_seqs, test_seqs = {}, {}
    for u, seq in user_seqs.items():
        if len(seq) > 1:
            train_seqs[u] = seq[:-1]
            test_seqs[u] = seq
    return ratings, train_seqs, test_seqs, user2idx, movie2idx, idx2movie

# ========== Advanced BERT-style Masking ==========
class MovieSequenceDataset(Dataset):
    def __init__(self, seq_dict, seq_len=10, mask_token=None):
        self.seqs = list(seq_dict.values())
        self.seq_len = seq_len
        self.mask_token = mask_token if mask_token is not None else 0

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][-self.seq_len:]
        if len(seq) < self.seq_len:
            seq = [0] * (self.seq_len - len(seq)) + seq
        # Simple: mask one random position
        pos = random.randrange(self.seq_len)
        label = seq[pos]
        masked_seq = seq.copy()
        masked_seq[pos] = self.mask_token
        return torch.tensor(masked_seq), torch.tensor(label)

# ========== Model Definitions ==========
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users+1, emb_size, padding_idx=0)
        self.item_emb = nn.Embedding(num_items+1, emb_size, padding_idx=0)
    def forward(self, u, v): return (self.user_emb(u) * self.item_emb(v)).sum(1)

class MiniBERTRec(nn.Module):
    def __init__(self, num_items, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(num_items+1, d_model, padding_idx=0)
        enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.head = nn.Linear(d_model, num_items+1)
    def forward(self, x):
        x = self.emb(x)  # No need to permute since batch_first=True
        out = self.transformer(x)
        return self.head(out)     # (batch, seq_len, vocab)

class MF_BERT_Hybrid(nn.Module):
    def __init__(self, num_items, pretrained_emb, d_model=None, nhead=4, num_layers=2):
        super().__init__()
        emb_size = pretrained_emb.size(1)
        self.emb = nn.Embedding(num_items+1, emb_size, padding_idx=0)
        self.emb.weight.data[1:] = pretrained_emb[1:]
        self.emb.weight.requires_grad = False
        d_model = d_model or emb_size
        enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.head = nn.Linear(d_model, num_items+1)
    def forward(self, x):
        x = self.emb(x)  # No need to permute since batch_first=True
        out = self.transformer(x)
        return self.head(out)     # (batch, seq_len, vocab)

# ========== Training Functions ==========
class RatingDataset(Dataset):
    def __init__(self, ratings_df):
        self.uids = ratings_df['uid'].values
        self.mids = ratings_df['mid'].values
        self.ratings = ratings_df['rating'].values.astype(np.float32)
    def __len__(self): return len(self.ratings)
    def __getitem__(self, idx): return self.uids[idx], self.mids[idx], self.ratings[idx]

def train_mf(model, ratings, batch_size=1024, epochs=5, lr=1e-3, device='cpu', log_dir=None):
    dataset = RatingDataset(ratings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device); opt=torch.optim.Adam(model.parameters(), lr=lr); loss_fn=nn.MSELoss()
    losses = []
    for e in range(epochs):
        epoch_losses=[]
        for u,v,y in tqdm(loader, desc=f"MF Epoch {e+1}"):
            u,v,y = u.to(device).long(), v.to(device).long(), y.to(device)
            loss = loss_fn(model(u,v), y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(loss.item())
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {e+1} MF loss: {mean_loss:.4f}")
        losses.append(mean_loss)
        if log_dir:
            log_metrics({'epoch': e+1, 'mf_loss': mean_loss}, os.path.join(log_dir, 'mf_train_log.csv'))
    if log_dir:
        plt.plot(range(1,epochs+1),losses,label='MF train')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.savefig(os.path.join(log_dir,'mf_train_loss.png'))
        plt.close()
    return model.cpu()

def train_bert(model, train_loader, val_seqs, epochs=10, lr=1e-3, device='cpu', log_dir=None):
    model.to(device); opt=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn=nn.CrossEntropyLoss()
    train_losses,val_losses=[],[]
    for e in range(epochs):
        model.train(); batch_train=[]
        for x,y in tqdm(train_loader, desc=f"BERT Epoch {e+1}"):
            x,y = x.to(device), y.to(device)
            logits = model(x)  # (batch, seq_len, vocab)
            # Use only last token for next-movie prediction
            logits = logits[:, -1, :]  # (batch, vocab)
            loss=loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            batch_train.append(loss.item())
        train_losses.append(np.mean(batch_train))
        # validation
        model.eval(); batch_val=[]
        for seq in val_seqs.values():
            inp=seq[:-1][-10:]
            if len(inp)<10: inp=[0]*(10-len(inp))+inp
            x=torch.tensor([inp]).to(device)
            logits = model(x)
            logits = logits[:, -1, :]
            y=torch.tensor([seq[-1]]).to(device)
            loss=loss_fn(logits, y)
            batch_val.append(loss.item())
        val_losses.append(np.mean(batch_val))
        if log_dir:
            log_metrics({'epoch': e+1, 'train_loss': train_losses[-1], 'val_loss': val_losses[-1]}, os.path.join(log_dir, 'bert_train_log.csv'))
    if log_dir:
        plt.plot(range(1,epochs+1),train_losses,label='train')
        plt.plot(range(1,epochs+1),val_losses,label='val')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.savefig(os.path.join(log_dir,'bert_train_val_loss.png'))
        plt.close()
    return model.cpu()

def train_hybrid(model, train_loader, val_seqs, total_epochs=10, freeze_ratio=0.3, lr=1e-3, device='cpu', log_dir=None):
    freeze_epochs = int(total_epochs * freeze_ratio)
    unfreeze_epochs = total_epochs - freeze_epochs
    model.emb.weight.requires_grad=False
    model = train_bert(model, train_loader, val_seqs, epochs=freeze_epochs, lr=lr, device=device, log_dir=log_dir)
    model.emb.weight.requires_grad=True
    model = train_bert(model, train_loader, val_seqs, epochs=unfreeze_epochs, lr=lr, device=device, log_dir=log_dir)
    return model

# ========== Evaluation ==========
def evaluate_bert(model, seq_dict, k=10, device='cpu', log_dir=None, exp_id=None, model_name=None):
    model.to(device).eval(); hits, ndcgs=[],[]
    for seq in seq_dict.values():
        if len(seq)<2: continue
        inp=seq[:-1][-10:];
        if len(inp)<10: inp=[0]*(10-len(inp))+inp
        x=torch.tensor([inp]).to(device); true_id=seq[-1]
        with torch.no_grad():
            logits = model(x)  # (1, seq_len, vocab)
            scores = logits[:, -1, :].squeeze().cpu().numpy()  # use last token
        topk=np.argsort(-scores)[:k]; hits.append(int(true_id in topk))
        true_rel=np.zeros_like(scores); true_rel[true_id]=1
        ndcgs.append(ndcg_score([true_rel],[scores],k=k))
    hit, ndcg = np.mean(hits), np.mean(ndcgs)
    print(f"Hit@{k}: {hit:.4f}, NDCG@{k}: {ndcg:.4f}")
    if log_dir and exp_id and model_name:
        log_metrics({'exp_id': exp_id, 'model': model_name, 'hit@k': hit, 'ndcg@k': ndcg}, os.path.join(log_dir, 'eval_metrics.csv'))
    return hit, ndcg

# ========== Recommendation ==========
def recommend(model, idx2movie, user_seq, top_k=5, device='cpu'):
    model.to(device).eval(); seq=user_seq[-10:]
    if len(seq)<10: seq=[0]*(10-len(seq))+seq
    x=torch.tensor([seq]).to(device)
    with torch.no_grad():
        logits = model(x)  # (1, seq_len, vocab)
        scores = logits[:, -1, :].squeeze()  # use last token
        probs=torch.softmax(scores,dim=0)
    vals, idxs=probs.topk(top_k)
    return [(idx2movie[i.item()],v.item()) for i,v in zip(idxs,vals)]

# ========== Main Logic ==========
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['bert','hybrid','both'],default='both')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--user',type=int)
    parser.add_argument('--skip-train',action='store_true')
    parser.add_argument('--device',default='cpu')
    # New MF/Hybrid hyperparameters
    parser.add_argument('--mf-epochs', type=int, default=5, help='Epochs for MF pretraining in hybrid')
    parser.add_argument('--mf-batch-size', type=int, default=1024, help='Batch size for MF pretraining')
    parser.add_argument('--mf-lr', type=float, default=1e-3, help='Learning rate for MF pretraining')
    parser.add_argument('--freeze-ratio', type=float, default=0.3, help='Fraction of hybrid epochs to freeze MF embeddings')
    parser.add_argument('--data-dir', type=str, default='ml-latest-small', help='Directory containing ratings.csv and movies.csv')
    args=parser.parse_args()

    exp_id = get_experiment_id()
    log_dir = os.path.join('model', f'exp_{exp_id}')
    os.makedirs(log_dir,exist_ok=True)
    ratings,train_seqs,test_seqs,user2idx,movie2idx,idx2movie=load_and_split(args.data_dir)
    idx2user={v:k for k,v in user2idx.items()}

    uid = user2idx.get(args.user, list(test_seqs.keys())[0]) if args.user else list(test_seqs.keys())[0]
    pickle.dump(test_seqs,open(os.path.join(log_dir,'test_seqs.pkl'),'wb'))
    save_json({'exp_id': exp_id, 'args': vars(args)}, os.path.join(log_dir, 'config.json'))

    models={}
    # BERT
    if not args.skip_train and args.model in ['bert','both']:
        ds=MovieSequenceDataset(train_seqs, seq_len=10)
        loader=DataLoader(ds,batch_size=64,shuffle=True)
        bert=MiniBERTRec(num_items=len(movie2idx))
        train_bert(bert,loader,test_seqs,epochs=args.epochs or 20,device=args.device, log_dir=log_dir)
        evaluate_bert(bert,test_seqs,device=args.device, log_dir=log_dir, exp_id=exp_id, model_name='bert')
        torch.save(bert.state_dict(),os.path.join(log_dir,'bert_model.pt')); models['bert']=bert
    elif args.model in ['bert','both']:
        bert=MiniBERTRec(num_items=len(movie2idx)); bert.load_state_dict(torch.load(os.path.join(log_dir,'bert_model.pt'),map_location=args.device)); models['bert']=bert
    # Hybrid
    if not args.skip_train and args.model in ['hybrid','both']:
        mf=MF(len(user2idx),len(movie2idx));
        mf=train_mf(mf,ratings,batch_size=args.mf_batch_size,epochs=args.mf_epochs,lr=args.mf_lr,device=args.device, log_dir=log_dir)
        emb=mf.item_emb.weight.data.clone(); hybrid=MF_BERT_Hybrid(num_items=len(movie2idx),pretrained_emb=emb)
        ds=MovieSequenceDataset(train_seqs, seq_len=10)
        loader=DataLoader(ds,batch_size=64,shuffle=True)
        hybrid=train_hybrid(hybrid,loader,test_seqs,total_epochs=args.epochs or 10,device=args.device, log_dir=log_dir, freeze_ratio=args.freeze_ratio)
        evaluate_bert(hybrid,test_seqs,device=args.device, log_dir=log_dir, exp_id=exp_id, model_name='hybrid')
        torch.save(hybrid.state_dict(),os.path.join(log_dir,'hybrid_model.pt')); models['hybrid']=hybrid
    elif args.model in ['hybrid','both']:
        hybrid=MF_BERT_Hybrid(num_items=len(movie2idx),pretrained_emb=torch.zeros(len(movie2idx)+1,32)); hybrid.load_state_dict(torch.load(os.path.join(log_dir,'hybrid_model.pt'),map_location=args.device)); models['hybrid']=hybrid

    movies_df=pd.read_csv(os.path.join(args.data_dir,'movies.csv'))
    # Print context
    train_ints=train_seqs.get(uid,[])
    print(f"Model input (train) sequence for user {idx2user.get(uid)} (total {len(train_ints)}):")
    for i in train_ints:
        row=movies_df[movies_df['movieId']==idx2movie[i]].iloc[0]
        print(f"  - {row['title']} [{row['genres']}]")
    full=test_seqs.get(uid,[])
    if len(full)>len(train_ints): row=movies_df[movies_df['movieId']==full[-1]].iloc[0]; print(f"\nHeld-out next item: {row['title']} [{row['genres']}]\n")
    # Recommendations
    for name,mdl in models.items():
        recs=recommend(mdl,idx2movie,test_seqs.get(uid,[]),device=args.device)
        print(f"Recommendations ({name}) for user {idx2user.get(uid)}:")
        for m_id,score in recs:
            row=movies_df[movies_df['movieId']==m_id].iloc[0]
            print(f"  - {row['title']} [{row['genres']}] (score: {score:.4f})")
    print('Done.')