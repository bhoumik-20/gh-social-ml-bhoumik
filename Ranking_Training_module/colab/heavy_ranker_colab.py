#This to not for deploy, just the training code.


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import joblib
import os

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Define the MMoE Architecture

# %%
class MMoEHeavyRanker(nn.Module):
    def __init__(self, input_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = 5 # CTR, Save, GH_Open, Dwell, Follow
        
        # 1. Shared Bottom Experts (2-layer MLP per expert)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU()
            ) for _ in range(self.num_experts)
        ])
        
        # 2. Gates (One for each of the 5 tasks)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.num_experts),
                nn.Softmax(dim=1)
            ) for _ in range(self.num_tasks)
        ])
        
        # 3. Task-Specific Heads (Inputs: 128-dim mixed expert vector)
        # Head 0: CTR (Binary -> Sigmoid)
        self.head_ctr = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        # Head 1: Save (Binary -> Sigmoid)
        self.head_save = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        # Head 2: GH Open (Binary -> Sigmoid)
        self.head_gh = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        # Head 3: Dwell Time (Regression -> ReLU for positive time)
        self.head_dwell = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU())
        # Head 4: Follow (Binary -> Sigmoid)
        self.head_follow = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        # Run through all experts. output shape: [batch_size, num_experts, 128]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Task 0: CTR
        gate_ctr = self.gates[0](x).unsqueeze(2) # [batch, num_experts, 1]
        mix_ctr = torch.sum(expert_outputs * gate_ctr, dim=1) # [batch, 128]
        out_ctr = self.head_ctr(mix_ctr).squeeze()
        
        # Task 1: Save
        gate_save = self.gates[1](x).unsqueeze(2)
        mix_save = torch.sum(expert_outputs * gate_save, dim=1)
        out_save = self.head_save(mix_save).squeeze()
        
        # Task 2: GH Open
        gate_gh = self.gates[2](x).unsqueeze(2)
        mix_gh = torch.sum(expert_outputs * gate_gh, dim=1)
        out_gh = self.head_gh(mix_gh).squeeze()
        
        # Task 3: Dwell Time
        gate_dwell = self.gates[3](x).unsqueeze(2)
        mix_dwell = torch.sum(expert_outputs * gate_dwell, dim=1)
        out_dwell = self.head_dwell(mix_dwell).squeeze()
        
        # Task 4: Follow
        gate_follow = self.gates[4](x).unsqueeze(2)
        mix_follow = torch.sum(expert_outputs * gate_follow, dim=1)
        out_follow = self.head_follow(mix_follow).squeeze()
        
        return out_ctr, out_save, out_gh, out_dwell, out_follow

# %% [markdown]
# ## 2. Data Loading & Preprocessing (Pure NumPy - Zero OOM)

# %%
import gc

class RankerDataset(Dataset):
    def __init__(self, user_embs, repo_embs, dense_feats, y_ctr, y_save, y_gh, y_dwell, y_follow):
        self.user_embs = user_embs
        self.repo_embs = repo_embs
        self.dense_feats = dense_feats
        self.y_ctr = y_ctr
        self.y_save = y_save
        self.y_gh = y_gh
        self.y_dwell = y_dwell
        self.y_follow = y_follow
        
    def __len__(self):
        return len(self.user_embs)
        
    def __getitem__(self, idx):
        # Concatenate on the fly to save RAM
        x_row = np.concatenate((self.user_embs[idx], self.repo_embs[idx], self.dense_feats[idx]))
        return (torch.tensor(x_row, dtype=torch.float32), 
                torch.tensor(self.y_ctr[idx], dtype=torch.float32), 
                torch.tensor(self.y_save[idx], dtype=torch.float32), 
                torch.tensor(self.y_gh[idx], dtype=torch.float32), 
                torch.tensor(self.y_dwell[idx], dtype=torch.float32), 
                torch.tensor(self.y_follow[idx], dtype=torch.float32))

print("Loading NPZ Data directly into memory...")
if not os.path.exists("training_data.npz"):
    print("❌ ERROR: training_data.npz not found.")
else:
    data = np.load("training_data.npz")
    
    # Extract features directly (already float32!)
    user_embs = data['user_embs']
    repo_embs = data['repo_embs']
    dense_features = data['dense_features']
    
    y_ctr = data['y_ctr']
    y_save = data['y_save']
    y_gh = data['y_gh']
    y_dwell = data['y_dwell']
    y_follow = data['y_follow']
    
    import json
    print("Scaling features...")
    scaler = StandardScaler()
    dense_features_scaled = scaler.fit_transform(dense_features).astype(np.float32)
    
    # Save safely as JSON instead of Pickle
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open('feature_scaler.json', 'w') as f:
        json.dump(scaler_params, f)
    
    # Use index-based splitting so we don't duplicate massive arrays in memory
    indices = np.arange(len(user_embs))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    train_dataset = RankerDataset(
        user_embs[train_idx], repo_embs[train_idx], dense_features_scaled[train_idx],
        y_ctr[train_idx], y_save[train_idx], y_gh[train_idx], y_dwell[train_idx], y_follow[train_idx]
    )
    val_dataset = RankerDataset(
        user_embs[val_idx], repo_embs[val_idx], dense_features_scaled[val_idx],
        y_ctr[val_idx], y_save[val_idx], y_gh[val_idx], y_dwell[val_idx], y_follow[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    input_dim = user_embs.shape[1] + repo_embs.shape[1] + dense_features.shape[1]
    print(f"Data ready. Input dim: {input_dim}, Train size: {len(train_dataset)}")

# %% [markdown]
# ## 3. Training Loop

# %%
model = MMoEHeavyRanker(input_dim=input_dim, num_experts=4).to(device)

# Multi-task Loss Weights
loss_bce = nn.BCELoss()
loss_mse = nn.MSELoss()

# Loss weights (Hyperparameters to balance tasks)
w_ctr, w_save, w_gh, w_dwell, w_follow = 1.0, 2.0, 1.5, 0.5, 5.0

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 5
print("Starting Training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        b_x, b_ctr, b_save, b_gh, b_dwell, b_fol = [t.to(device) for t in batch]
        
        optimizer.zero_grad()
        p_ctr, p_save, p_gh, p_dwell, p_fol = model(b_x)
        
        # Calculate individual losses
        l_ctr = loss_bce(p_ctr, b_ctr)
        l_save = loss_bce(p_save, b_save)
        l_gh = loss_bce(p_gh, b_gh)
        l_dwell = loss_mse(p_dwell, b_dwell)
        l_fol = loss_bce(p_fol, b_fol)
        
        # Weighted sum loss
        loss = (w_ctr * l_ctr) + (w_save * l_save) + (w_gh * l_gh) + (w_dwell * l_dwell) + (w_follow * l_fol)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Validation AUCs
    model.eval()
    val_preds = {'ctr':[], 'save':[], 'gh':[], 'fol':[]}
    val_trues = {'ctr':[], 'save':[], 'gh':[], 'fol':[]}
    
    with torch.no_grad():
        for batch in val_loader:
            b_x, b_ctr, b_save, b_gh, b_dwell, b_fol = [t.to(device) for t in batch]
            p_ctr, p_save, p_gh, p_dwell, p_fol = model(b_x)
            
            val_preds['ctr'].extend(p_ctr.cpu().numpy())
            val_trues['ctr'].extend(b_ctr.cpu().numpy())
            val_preds['save'].extend(p_save.cpu().numpy())
            val_trues['save'].extend(b_save.cpu().numpy())
            val_preds['fol'].extend(p_fol.cpu().numpy())
            val_trues['fol'].extend(b_fol.cpu().numpy())

    try:
        auc_ctr = roc_auc_score(val_trues['ctr'], val_preds['ctr'])
        auc_save = roc_auc_score(val_trues['save'], val_preds['save'])
        auc_fol = roc_auc_score(val_trues['fol'], val_preds['fol'])
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} | AUC: CTR={auc_ctr:.3f}, Save={auc_save:.3f}, Follow={auc_fol:.3f}")
    except ValueError:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

# %% [markdown]
# ## 4. Export the Model

# %%
torch.save(model.state_dict(), "heavy_ranker.pt")
print("✅ Training Complete! Model saved as 'heavy_ranker.pt'.")
print("Download 'heavy_ranker.pt' and 'feature_scaler.json' for your backend.")
