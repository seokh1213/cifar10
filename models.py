import torch
import torch.nn as nn
from torch.functional import F


class BaseNetwork(nn.Module):
    def train_process(model, optimizer, criterion, dataloader, device, verbose=True, print_iter=10):
        if verbose:
            print('Start Training.')
        model.train()
        running_loss = 0.0
        running_corrected = 0
        total_size = 0
        for iter, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()

            logits, preds = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            corrected = torch.sum(labels == preds).item()
            running_corrected += corrected
            total_size += labels.size(0)

            if verbose and iter % print_iter == 0:
                acc = corrected / labels.size(0)
                print(f'{total_size}/{len(dataloader.dataset)} Train Loss: {loss} Acc: {acc}')
        return running_loss / (iter + 1), running_corrected / total_size

    def valid_process(model, criterion, dataloader, device, verbose=True, print_iter=10):
        if verbose:
            print('Start Validating.')
        model.eval()
        running_loss = 0.0
        running_corrected = 0
        total_size = 0
        for iter, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.squeeze().to(device)
            with torch.no_grad():
                logits, preds = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item()

            corrected = torch.sum(labels == preds).item()
            running_corrected += corrected
            total_size += labels.size(0)

            if verbose and iter % print_iter == 0:
                acc = corrected / labels.size(0)
                print(f'\r{total_size}/{len(dataloader.dataset)} Valid Loss: {loss} Acc: {acc}', end='')
        return running_loss / (iter + 1), running_corrected / total_size


class NaiveNetwork(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5, 5))  # 3x32x32 -> 8x28x28
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5))  # 8x14x14 -> 16x10x10
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.pool(self.conv1(x))  # 3x32x32 -> 8x28x28 -> 8x14x14
        x = self.pool(self.conv2(x))  # 8x14x14 -> 16x5x5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits, logits.max(dim=1)[1]
