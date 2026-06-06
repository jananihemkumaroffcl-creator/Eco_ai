# eco_train.py - Optimized for carbon savings

import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from codecarbon import EmissionsTracker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 120
TARGET_ACCURACY = 0.97


# ================= DATA =================
def load_data():

    data = load_breast_cancer()

    X = StandardScaler().fit_transform(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test


# ================= MODEL =================
class EcoNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)


# ================= TRAIN =================
def train():

    tracker = EmissionsTracker(
        project_name="EcoAI",
        output_file="energy_log.csv",
        output_dir="data",
        measure_power_secs=1
    )

    tracker.start()

    X_train, X_test, y_train, y_test = load_data()

    batch_size = 64

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=256
    )

    model = EcoNet().to(DEVICE)

    # ----- PRUNING -----
    import torch.nn.utils.prune as prune

    prune.l1_unstructured(model.net[0], name="weight", amount=0.5)
    prune.l1_unstructured(model.net[2], name="weight", amount=0.3)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_acc = 0
    patience = 8
    patience_counter = 0

    log_rows = []
    prev_energy = 0.0
    prev_emissions = 0.0

    for epoch in range(MAX_EPOCHS):

        model.train()

        # ----- Dynamic batch sizing -----
        cpu_load = psutil.cpu_percent(interval=0.1)

        if cpu_load > 80:
            batch_size = 32
        else:
            batch_size = 64

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )

        # ----- Training -----
        for xb, yb in train_loader:

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(xb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ----- Evaluation -----
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for xb, yb in test_loader:

                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                preds = model(xb).argmax(dim=1)

                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total

        print(f"[Eco] Epoch {epoch+1} | Acc = {acc:.4f}")

        # ----- Update CodeCarbon -----
        tracker.flush()

        total_energy = float(tracker._total_energy.kWh)
        total_emissions = float(tracker._total_emissions)

        log_rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "energy_consumed": total_energy - prev_energy,
            "emissions": total_emissions - prev_emissions,
            "duration": epoch + 1
        })

        prev_energy = total_energy
        prev_emissions = total_emissions

        df = pd.DataFrame(log_rows)

        df.to_csv("data/eco_epoch_log.csv", index=False)

        # ----- Early stopping -----
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1

        if acc >= TARGET_ACCURACY or patience_counter >= patience:
            print("♻ Early stopping to save energy")
            break


    # ----- Save optimized model -----
    model.cpu()
    torch.save(model.state_dict(), "models/eco_model.pth")

    emissions = tracker.stop()

    print("🌱 Eco training done. CO2:", emissions)


# ================= RUN =================
if __name__ == "__main__":
    train()