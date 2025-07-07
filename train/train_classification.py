import torch
import torch.nn as nn
import torch.optim as optim
# ✅ Correct (package-relative import)
from models.classifier import ImageClassifier
from data.dataset_loader import get_dataloaders


def train(cfg):
    model = ImageClassifier(cfg['model']).to(cfg['training']['device'])
    train_loader, val_loader = get_dataloaders(cfg['training']['batch_size'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    for epoch in range(cfg['training']['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg['training']['device']), y.to(cfg['training']['device'])
            loss = criterion(model(x), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), cfg['training']['save_path'])
