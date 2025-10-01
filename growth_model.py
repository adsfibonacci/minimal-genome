import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

seed = 2483
np.random.seed(seed)

DATA = 'data/'

n_genes = 1968
def read_essential_knockouts():
    df = pd.read_csv(DATA + 'essential_entergenic.csv')
    oralgen_id = [i - 1 for i in df['Oralgen Gene ID'].tolist()]
    # print(oralgen_id)
    singletons = np.ones((n_genes, n_genes)) - np.eye(n_genes)
    response = np.ones(n_genes)
    response[oralgen_id] = 0
    return singletons, response


X, y = read_essential_knockouts()
X = X.astype(np.float32)
y = y.astype(np.int64)

class GrowthNN(nn.Module):
    def __init__(self, n_genes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
all_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}")
    
    # Split
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Torch tensors
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # New model each fold
    model = GrowthNN(n_genes)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Track history for this fold
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

    # Training loop
    for epoch in range(20):
        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val)
            val_loss = criterion(val_probs, y_val).item()
            val_preds = (val_probs > 0.5).float()
            val_acc = accuracy_score(y_val.numpy(), val_preds.numpy())

        # Save history
        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:02d}: "
              f"Train Loss={train_loss.item():.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Append history for this fold
    all_histories.append(pd.DataFrame(history))

# --- After training ---
# Concatenate all fold histories (multi-index: fold, epoch)
histories_df = pd.concat(all_histories, keys=range(1, len(all_histories)+1), names=["fold", "row"])
print("\nFinal CV history shape:", histories_df.shape)

# Example: mean validation accuracy across folds per epoch
mean_val_acc = histories_df.groupby("epoch")["val_acc"].mean()
print("\nMean val acc per epoch:\n", mean_val_acc)
