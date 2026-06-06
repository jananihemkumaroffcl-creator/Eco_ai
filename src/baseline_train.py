# baseline_train.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from codecarbon import EmissionsTracker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150

def load_data():
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )

class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

def train():
    tracker = EmissionsTracker(
        project_name="Baseline",
        output_file="baseline_energy_log.csv",
        output_dir="data",
        measure_power_secs=0.1
    )
    tracker.start()

    X_train, X_test, y_train, y_test = load_data()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

    model = BigNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        print(f"[Baseline] Epoch {epoch+1}")
        time.sleep(1)

    emissions = tracker.stop()
    print("🔥 Baseline done. CO2:", emissions)

if __name__ == "__main__":
    train()