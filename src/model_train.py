import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Configuration

GESTURES = ["fist", "open_hand", "pinch", "peace", "thumbs_up"]
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#  Dataset
class GestureDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)



# Load Data From .npy Files
X, y = [], []

for label, gesture in enumerate(GESTURES):
    folder = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            sample = np.load(os.path.join(folder, file))
            X.append(sample)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Loaded dataset:", X.shape, y.shape)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

train_ds = GestureDataset(X_train, y_train)
test_ds = GestureDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)


# Initialize Model
class GestureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(GESTURES))
        )

    def forward(self, x):
        return self.net(x)


model = GestureModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")


# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

acc = correct / total
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")


# Save the Model
torch.save(model.state_dict(), "jarvis.pth")
print("Jarvis has been built...")
