import numpy as np
import random
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from nltk_utils import bag_of_words, tokenize, stem, normalize_tokens
from model import NeuralNet

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        # normalize then extend
        w = normalize_tokens(w)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words")

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = X_train.shape[1]
hidden_size = 16
output_size = len(tags)

print("input_size", input_size, "output_size", output_size)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.x_data = torch.from_numpy(X).to(dtype=torch.float32)
        self.y_data = torch.from_numpy(y).to(dtype=torch.long)
        self.n_samples = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset(X_train, y_train)

# train/val split
val_ratio = 0.1 if len(dataset) >= 10 else 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Remove verbose=True for compatibility
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=20, factor=0.5
)

best_val_loss = float('inf')
patience = 80
stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * words.size(0)

    # validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for (vwords, vlabels) in val_loader:
            vwords = vwords.to(device)
            vlabels = vlabels.to(device)
            outputs = model(vwords)
            loss_v = criterion(outputs, vlabels)
            val_loss += loss_v.item() * vwords.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()

    avg_train_loss = running_loss / train_size
    avg_val_loss = val_loss / max(1, val_size)
    val_acc = correct / max(1, total)

    scheduler.step(avg_val_loss)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        # get current LR
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'TrainLoss: {avg_train_loss:.4f}, '
              f'ValLoss: {avg_val_loss:.4f}, '
              f'ValAcc: {val_acc:.4f}, '
              f'LR: {lr:.6f}')

    # early stopping
    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        stopping_counter = 0
        # save best model
        data = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "all_words": all_words,
            "tags": tags
        }
        torch.save(data, "data.pth")
    else:
        stopping_counter += 1
        if stopping_counter >= patience:
            print("Early stopping triggered.")
            break

print("Training finished. Best val loss:", best_val_loss)

# ensure data.pth exists (already saved when val improved)
if not os.path.exists("data.pth"):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    torch.save(data, "data.pth")
    print("Saved final model to data.pth")
