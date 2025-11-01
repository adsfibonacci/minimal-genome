import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Custom datareader from pandas dataframes
# View default DATA parameters defined in the following functions
from datareader import read_ko, read_kd
# -----------------------
# Config / hyperparams
# -----------------------
intergenic = False
n_genes = 1968 if not intergenic else 2435
n_heads = 4
lr = 1e-6
epochs = 25
n_splits = 5
seed = 4

torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_single, y_single = read_ko(intergenic)
X_opmod, y_opmod = read_kd(intergenic, np.where(y_single == 0)[0])

X = np.vstack([X_single, X_opmod])
y = np.concatenate([y_single, y_opmod])
y_strat = np.round(y).astype(np.int64)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

class GenomeCNN(nn.Module):
    def __init__(self, n_genes=1968):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=12, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=12, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # flatten size = 64 * 489 = 31296
        self.fc = nn.Sequential(
            nn.Linear(64 * 489, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_strat)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    
    # Data split
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size)
    
    # Model, loss, optimizer
    model = GenomeCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()  # handles soft labels
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Store metrics for this fold
    fold_records = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            pass
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                preds_prob = torch.sigmoid(preds)
                val_correct += ((preds_prob > 0.5) == (yb > 0.5)).sum().item()
                val_total += yb.size(0)
                pass
            pass
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        fold_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        print(f"Epoch {epoch+1:02d}/{epochs} "
              f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} ValAcc={val_acc:.4f}")
        pass
    fold_histories.append(pd.DataFrame(fold_records))
    pass


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
plt.savefig("cnn_loss_acc.png")
# plt.show()
plt.close()
