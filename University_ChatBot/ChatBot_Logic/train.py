import numpy as np
import random
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from nltk_utils import bag_of_words, tokenize, stem, normalize_tokens
from model import NeuralNet

# ==========================================================
# FIX 1: Always use THIS folder for all file paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_PATH = os.path.join(BASE_DIR, "data.pth")

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================================
# Load intents.json correctly
# ==========================================================
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        tokens = normalize_tokens(tokens)
        all_words.extend(tokens)
        xy.append((tokens, tag))

ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words")

# Training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

# ==========================================================
# Model hyperparameters
# ==========================================================
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(tags)

print("input_size:", input_size, "output_size:", output_size)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()
        self.n_samples = len(X)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset(X_train, y_train)

# Train / validation split
val_ratio = 0.1 if len(dataset) >= 10 else 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20
)

best_val_loss = float("inf")
patience = 80
early_stop_counter = 0

# ==========================================================
# Training loop
# ==========================================================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * words.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for vwords, vlabels in val_loader:
            vwords, vlabels = vwords.to(device), vlabels.to(device)

            outputs = model(vwords)
            loss_v = criterion(outputs, vlabels)

            val_loss += loss_v.item() * vwords.size(0)

            _, predicted = torch.max(outputs, 1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()

    avg_train_loss = train_loss / train_size
    avg_val_loss = val_loss / max(1, val_size)
    val_acc = correct / max(1, total)

    scheduler.step(avg_val_loss)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}]  "
            f"TrainLoss: {avg_train_loss:.4f}  "
            f"ValLoss: {avg_val_loss:.4f}  "
            f"ValAcc: {val_acc:.4f}  "
            f"LR: {current_lr:.6f}"
        )

    # Early stopping
    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        early_stop_counter = 0

        # ==========================================================
        # FIX 2: Always save model inside ChatBot_Logic folder
        # ==========================================================
        torch.save({
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "all_words": all_words,
            "tags": tags
        }, MODEL_PATH)

    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training finished. Best validation loss:", best_val_loss)

# Ensure final model is saved
if not os.path.exists(MODEL_PATH):
    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
