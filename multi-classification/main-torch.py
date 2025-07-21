import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 5
    NUM_CLASSES = 10

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    base_model = models.mobilenet_v2(pretrained=True)

    base_model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base_model.last_channel, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES)
    )

    base_model = base_model.to(device)

    for param in base_model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        base_model.train()
        total_loss = 0
        correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = base_model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        acc = correct / len(train_dataset)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} tamamlandı - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

    torch.save(base_model.state_dict(), "multi_classification_model.pth")
    print("✅ Model saved as multi_classification_model.pth")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()
