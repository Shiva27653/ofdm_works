# imports
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = r"C:\Users\Vedant\Desktop\ofdm_works\Gauss"

# check dataset files
mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
print(f"Found {len(mat_files)} .mat files:")
for f in sorted(mat_files):
    print(f"  {f}")

# inspect nested .mat structure
data = sio.loadmat(os.path.join(DATA_DIR, 'QPSK.mat'))
dataset = data['dataset']
print(f"Outer shape: {dataset.shape}, dtype: {dataset.dtype}")

entry = dataset[0, 0]
print(f"\nFirst entry type: {type(entry)}")
print(f"First entry dtype: {entry.dtype}")
print(f"First entry shape: {entry.shape}")

for i in range(min(entry.shape[1], 5)):
    inner = entry[0, i]
    if isinstance(inner, np.ndarray):
        print(f"  sub[{i}] shape={inner.shape} dtype={inner.dtype}")

# extraction helper
def extract_sample(entry):
    """Extract (2,1024) signal and snr from a nested mat entry."""
    signal = None
    snr = None

    def _recurse(obj):
        nonlocal signal, snr
        if not isinstance(obj, np.ndarray):
            return
        if obj.shape == (2, 1024) and obj.dtype in (np.float64, np.float32):
            signal = obj
            return
        if obj.shape == (1, 1) and obj.dtype in (np.int16, np.int32, np.int64):
            snr = int(obj[0, 0])
            return
        if obj.dtype == object:
            for item in obj.flat:
                _recurse(item)
                if signal is not None and snr is not None:
                    return

    _recurse(entry)
    return signal, snr

# quick test
for fname in ['QPSK.mat', '16QAM.mat']:
    d = sio.loadmat(os.path.join(DATA_DIR, fname))['dataset']
    print(f"\n{fname}:")
    for i in [0, 1, 100]:
        sig, snr = extract_sample(d[0, i])
        print(f"  sample[{i}] → signal shape={sig.shape}, snr={snr}")

# visual sanity checks
fig, axes = plt.subplots(2, 3, figsize=(14, 6))
sample_files = ['BPSK.mat', 'QPSK.mat', '8PSK.mat', '16QAM.mat', '64QAM.mat', '256QAM.mat']

for idx, fname in enumerate(sample_files):
    d = sio.loadmat(os.path.join(DATA_DIR, fname))['dataset']
    sig, snr = extract_sample(d[0, 2000])
    ax = axes[idx // 3, idx % 3]
    ax.plot(sig[0, :200], label='I', alpha=0.8, linewidth=0.7)
    ax.plot(sig[1, :200], label='Q', alpha=0.8, linewidth=0.7)
    ax.set_title(f"{fname.replace('.mat','')} (SNR={snr})")
    ax.legend(fontsize=7)
    ax.set_xlabel('Sample')

plt.tight_layout()
plt.show()

sig_qpsk, _ = extract_sample(sio.loadmat(os.path.join(DATA_DIR, 'QPSK.mat'))['dataset'][0, 5000])
plt.figure(figsize=(4, 4))
plt.scatter(sig_qpsk[0], sig_qpsk[1], s=1, alpha=0.5)
plt.title('IQ scatter - QPSK')
plt.xlabel('I')
plt.ylabel('Q')
plt.axis('equal')
plt.show()

# extract all samples
signals = []
labels = []
snrs = []

for fname in sorted(mat_files):
    class_name = fname.replace('.mat', '')
    d = sio.loadmat(os.path.join(DATA_DIR, fname))['dataset']
    count = 0
    for i in range(d.shape[1]):
        sig, snr = extract_sample(d[0, i])
        if sig is not None:
            signals.append(sig)
            labels.append(class_name)
            snrs.append(snr if snr is not None else -999)
            count += 1
    print(f"{class_name}: extracted {count} samples")

print(f"\nTotal: {len(signals)} samples")

# convert to arrays
X = np.stack(signals).astype(np.float32)
snr_arr = np.array(snrs)
del signals, snrs

le = LabelEncoder()
y = le.fit_transform(labels)
del labels

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {le.classes_}")
print(f"\nSamples per class:")
for cls, name in enumerate(le.classes_):
    print(f"  {name}: {np.sum(y == cls)}")

print(f"\nSNR distribution:")
unique_snrs, snr_counts = np.unique(snr_arr, return_counts=True)
for s, c in zip(unique_snrs, snr_counts):
    print(f"  SNR={s:+d} dB: {c} samples")

# preprocessing and split
# per-sample z-score so model learns shape not amplitude
X_mean = X.mean(axis=(1, 2), keepdims=True)
X_std = X.std(axis=(1, 2), keepdims=True) + 1e-8
X_norm = (X - X_mean) / X_std

X_flat = X_norm.reshape(X_norm.shape[0], -1)
X_2ch = X_norm

X_flat_train, X_flat_val, y_train, y_val = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y)
X_2ch_train, X_2ch_val, _, _ = train_test_split(
    X_2ch, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_flat_train.shape[0]} samples, Val: {X_flat_val.shape[0]} samples")
print(f"Flat shape: train={X_flat_train.shape}, val={X_flat_val.shape}")
print(f"2ch shape:  train={X_2ch_train.shape}, val={X_2ch_val.shape}")

batch_size = 256

def make_loaders(X_tr, X_v, y_tr, y_v):
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_v), torch.LongTensor(y_v))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl

flat_train_dl, flat_val_dl = make_loaders(X_flat_train, X_flat_val, y_train, y_val)
ch2_train_dl, ch2_val_dl = make_loaders(X_2ch_train, X_2ch_val, y_train, y_val)

print(f"\nDataLoaders ready (batch_size={batch_size})")
print(f"  Flat: {len(flat_train_dl)} train batches, {len(flat_val_dl)} val batches")
print(f"  2ch:  {len(ch2_train_dl)} train batches, {len(ch2_val_dl)} val batches")

# training loop and mlp
assert torch.cuda.is_available(), "GPU not found! Enable GPU in Kaggle: Settings → Accelerator → T4x2"
device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {device}\n")

def train_model(model, train_dl, val_dl, epochs=20, lr=1e-3):
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc = eval_model(model, val_dl)
            print(f"  Epoch {epoch+1:2d} | loss={total_loss/total:.4f} | train_acc={correct/total:.4f} | val_acc={val_acc:.4f}")

    return model

def eval_model(model, val_dl):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )
    def forward(self, x):
        return self.net(x)

print("Training MLP...")
mlp = train_model(MLPClassifier(), flat_train_dl, flat_val_dl, epochs=30)

# mlp evaluation
def get_predictions(model, val_dl):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    return np.array(all_preds), np.array(all_labels)

OUT_DIR = '/kaggle/working/outputs'
os.makedirs(OUT_DIR, exist_ok=True)

mlp_preds, mlp_true = get_predictions(mlp, flat_val_dl)
mlp_acc = accuracy_score(mlp_true, mlp_preds)

print(f"MLP Validation Accuracy: {mlp_acc:.4f}\n")
print(classification_report(mlp_true, mlp_preds, target_names=le.classes_))

cm = confusion_matrix(mlp_true, mlp_preds)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(le.classes_, rotation=45)
ax.set_yticklabels(le.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'MLP Confusion Matrix (acc={mlp_acc:.2%})')
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/mlp_confusion_matrix.png', dpi=150)
plt.show()

# 1d cnn
# no batchnorm — breaks with DataParallel (per-gpu stats desync)
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

print("Training 1D CNN...")
cnn = train_model(CNN1D(), ch2_train_dl, ch2_val_dl, epochs=90, lr=3e-4)

# cnn evaluation
cnn_preds, cnn_true = get_predictions(cnn, ch2_val_dl)
cnn_acc = accuracy_score(cnn_true, cnn_preds)

print(f"CNN Validation Accuracy: {cnn_acc:.4f}\n")
print(classification_report(cnn_true, cnn_preds, target_names=le.classes_))

cm_cnn = confusion_matrix(cnn_true, cnn_preds)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_cnn, cmap='Greens')
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(le.classes_, rotation=45)
ax.set_yticklabels(le.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'1D CNN Confusion Matrix (acc={cnn_acc:.2%})')
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm_cnn[i, j], ha='center', va='center', fontsize=8)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cnn_confusion_matrix.png', dpi=150)
plt.show()

# cnn + lstm
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)       # (batch, seq=64, feat=128)
        _, (hn, _) = self.lstm(x)
        x = hn.squeeze(0)
        return self.classifier(x)

print("Training CNN+LSTM...")
cnn_lstm = train_model(CNNLSTM(), ch2_train_dl, ch2_val_dl, epochs=60, lr=5e-4)

# cnn+lstm evaluation
cl_preds, cl_true = get_predictions(cnn_lstm, ch2_val_dl)
cl_acc = accuracy_score(cl_true, cl_preds)

print(f"CNN+LSTM Validation Accuracy: {cl_acc:.4f}\n")
print(classification_report(cl_true, cl_preds, target_names=le.classes_))

cm_cl = confusion_matrix(cl_true, cl_preds)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_cl, cmap='Oranges')
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(le.classes_, rotation=45)
ax.set_yticklabels(le.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'CNN+LSTM Confusion Matrix (acc={cl_acc:.2%})')
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm_cl[i, j], ha='center', va='center', fontsize=8)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cnn_lstm_confusion_matrix.png', dpi=150)
plt.show()

# comparison
def count_params(model):
    m = model.module if hasattr(model, 'module') else model
    return sum(p.numel() for p in m.parameters())

results = pd.DataFrame({
    'Model': ['MLP', '1D CNN', 'CNN+LSTM'],
    'Val Accuracy': [f'{mlp_acc:.4f}', f'{cnn_acc:.4f}', f'{cl_acc:.4f}'],
    'Parameters': [f'{count_params(mlp):,}', f'{count_params(cnn):,}', f'{count_params(cnn_lstm):,}']
})
print(results.to_string(index=False))
print("\nObservations:")
print(f"  - MLP baseline: {mlp_acc:.2%}")
print(f"  - 1D CNN: {cnn_acc:.2%}")
print(f"  - CNN+LSTM: {cl_acc:.2%}")

results.to_csv(f'{OUT_DIR}/model_comparison.csv', index=False)
print(f"\nSaved to {OUT_DIR}/model_comparison.csv")
