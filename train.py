import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pathlib
import time

# ───────────────────────── Device ─────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────── Data pipeline ─────────────────────
transform = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_ds = datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ──────────────────────── Model ───────────────────────────
model = nn.Sequential(
    nn.Flatten(),              #  1×28×28 → 784
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# ───────────── Loss, optimiser, hyper-params ─────────────
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

# ──────────────── Containers for plots ───────────────────
train_losses, train_accs, test_accs = [], [], []

# ───────────────────────── Training ──────────────────────
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds      = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total
    train_losses.append(train_loss)
    train_accs .append(train_acc)

    # ─────────────── Evaluate on test set ────────────────
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            _, preds     = torch.max(outputs, 1)
            total       += labels.size(0)
            correct     += (preds == labels).sum().item()

    test_acc = 100 * correct / total
    test_accs.append(test_acc)

    print(f"Epoch {epoch:02}/{num_epochs} | "
          f"Train Loss {train_loss:.4f} | "
          f"Train Acc {train_acc:.2f}% | "
          f"Test Acc {test_acc:.2f}%")

# ────────────────────────── Plot ─────────────────────────
ts = time.strftime("%Y%m%d-%H%M%S")
plot_path = pathlib.Path("training_metrics.png")

plt.figure(figsize=(6,4))
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs,  label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Fashion-MNIST accuracy over epochs")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\n\u2713 Training complete. Metric plot saved → {plot_path.resolve()}")
