import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from ast import literal_eval

# -----------------------
# Config / hyperparams
# -----------------------
DATA = "./data/" # Root path, change based off computer file structure 
n_genes = 1968
embed_dim = 32
n_heads = 4
transformer_layers = 1
hidden_clf = 32
dropout = 0.5
lr = 1e-3
epochs = 20
batch_size = 64
n_splits = 5
seed = 4

torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Single genes are removed from the genome in the desired way.
There are 1968 genes, so there are 1969 rows.
Each row is a gene with a -1 indicating the gene is not present and a 1 indicating it is present.
The response is a 1 if the genome grows and a 0 if it does not. 
Returns a 1969 * 1968 matrix and a 1969 binary response vector. 
"""
def read_essential_knockouts():
    df = pd.read_csv(DATA + 'singles/essential_entergenic.csv')
    oralgen_id = df['Oralgen Gene ID'].tolist()
    singletons = np.ones((n_genes + 1, n_genes), dtype=np.int64)
    singletons[1:] -= 2 * np.eye(n_genes, dtype=np.int64)
    response = np.ones(n_genes + 1, dtype=np.int64)
    response[oralgen_id] = 0
    return singletons, response, oralgen_id

"""
Entire operons are downregulated in the genome.
Since some operons are single genes, we ignore those since the true list of knockouts is present in the matrix above.
Downregulated operons with multiple genes are treated as an entire operon knockout, with genes in the operon having a -1 in the row's.
Response is binary with 1 if the genome grows and a 0 if not.
Returns a 368 x 1968 matrix and a 368 binary response vector. 
"""
def read_essential_knockdowns():
    df = pd.read_csv(DATA + 'opmod/opmod_growth.csv')
    df['Operon Map'] = [literal_eval(df['Operon Map'][i]) for i in range(len(df))]
    df = df[[len(mapping) > 1 for mapping in df['Operon Map']]]
    df = df[df['Payload_Name'].str.contains('KD')].reset_index(drop=True)

    kd_df = df[(df['colony_growth'] != 'Y') | (df['liquid_growth'] != 'Y')]
    kd_df.to_csv(DATA + 'opmod/knockdown_multiple.csv', index=False)

    oralgen_id = kd_df['Operon Map'].tolist()
    masks = df['Operon Map'].tolist()
    
    genomes = np.ones((len(df), n_genes), dtype=np.int64)
    for i, mask in enumerate(masks):
        mask = [j-1 for j in mask]
        genomes[i][mask] = -1
    response = np.ones(len(df), dtype=np.int64)
    response[kd_df.index] = 0
    return genomes, response, oralgen_id

X_single, y_single, essential_single = read_essential_knockouts()
X_opmod, y_opmod, essential_opmod = read_essential_knockdowns()

X = np.vstack([X_single, X_opmod])
y = np.concatenate([y_single, y_opmod])

print(f"X shape: {X.shape}, Y shape: {y.shape}")
print(f"Essential single knockouts: {len(essential_single)}, Essential downregulated opmods: {len(essential_opmod)}")
print(f"Embed dim: {embed_dim}, n_heads: {n_heads}, transformer_layers: {transformer_layers}, hidden_clf: {hidden_clf}, dropout: {dropout}")

np.save(DATA + "operons.npy", X)
np.save(DATA + "response.npy", y)

X_idx_all = [list(np.where(row == -1)[0]) for row in X]
print(X[0:10, 0:10])

"""
Custom Dataset loader class templated from PyTorch Dataset Class
"""
class KnockoutSetDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = np.array(y, dtype=np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

"""
Padding function for the Dataloader that makes sure ensure the vector inputs are same dimension. 
"""
def collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    lengths = [len(l) for l in X_batch]
    max_len = max(max(lengths), 1)
    pad = n_genes
    seqs = np.full((len(X_batch), max_len), pad, dtype=np.int64)
    for i, l in enumerate(X_batch):
        if len(l) > 0:
            seqs[i, :len(l)] = l
    seqs = torch.tensor(seqs, dtype=torch.long)
    y_batch = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1)
    pad_mask = (seqs == pad)  # True where padded
    return seqs, y_batch, pad_mask


"""
Pass the padded genome sequences to the transformer to train on.
The sets of knockouts are padded with the n_genes value since the column numbers range from 0 to n_genes - 1.
The knockout set is padded to the maximum length of knockouts. 
"""
class SetTransformerClassifier(nn.Module):
    def __init__(self, n_genes, embed_dim, n_heads, num_layers, hidden_clf, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(n_genes + 1, embed_dim, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_clf),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_clf, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        if pad_mask is not None:
            x = self.transformer(x, src_key_padding_mask=pad_mask)
        else:
            x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# -----------------------
# Training
# -----------------------
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
dataset = KnockoutSetDataset(X_idx_all, y)
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    print(f"\n=== Fold {fold} ===")
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = SetTransformerClassifier(n_genes=n_genes, embed_dim=embed_dim,
                                     n_heads=n_heads, num_layers=transformer_layers,
                                     hidden_clf=hidden_clf, dropout=dropout,
                                     pad_idx=n_genes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"epoch":[], "train_loss":[], "val_loss":[], "val_acc":[]}

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for seqs, labels, pad_mask in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            pad_mask = pad_mask.to(device)
            optimizer.zero_grad()
            preds = model(seqs, pad_mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for seqs, labels, pad_mask in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                pad_mask = pad_mask.to(device)
                preds = model(seqs, pad_mask)
                loss = criterion(preds, labels)
                val_losses.append(loss.item())
                preds_all.append((preds.cpu().numpy() > 0.5).astype(int))
                labels_all.append(labels.cpu().numpy().astype(int))
        val_loss = float(np.mean(val_losses))
        preds_all = np.vstack(preds_all)
        labels_all = np.vstack(labels_all)
        val_acc = accuracy_score(labels_all, preds_all)

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)        

    fold_histories.append(pd.DataFrame(history))

# Combine histories
histories_df = pd.concat(fold_histories, keys=range(1, len(fold_histories)+1), names=["fold","row"])
print("\nTraining complete. Histories shape:", histories_df.shape)

# Average across folds by epoch
avg_history = histories_df.groupby("epoch").mean()
print("\nAverage history per epoch:\n", avg_history)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(avg_history['train_loss'], label='Train Loss')
plt.plot(avg_history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss per Epoch")

plt.subplot(1,2,2)
plt.plot(avg_history['val_acc'], label='Val Accuracy', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.show()
