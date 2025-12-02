import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dir = "dataset/datasets/thermal_classification_cropped"
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(train_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

num_epochs = 8
best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("Training started...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}  Batch {batch_idx+1}/{len(train_loader)}  Loss: {loss.item():.4f}")

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)

    history["train_loss"].append(avg_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_dir, "thermal_model_best.pth"))

    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
    print(f"Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%  Best Val Acc: {best_val_acc:.2f}%")
    print("-" * 60)

total_time = time.time() - start_time
print(f"Training completed in {total_time/60:.2f} minutes")

final_model_path = os.path.join(save_dir, "thermal_model_final.pth")
torch.save(model.state_dict(), final_model_path)

print("Model training summary:")
for i in range(num_epochs):
    print(f"Epoch {i+1}: Train Acc {history['train_acc'][i]:.2f}% | Val Acc {history['val_acc'][i]:.2f}%")

print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Final model saved to {final_model_path}")
