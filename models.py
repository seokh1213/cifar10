import torch
import torch.nn as nn
from torch.functional import F
import timm


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

            logits = model(images)
            _, preds = logits.max(dim=1)
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
                logits = model(images)
                _, preds = logits.max(dim=1)
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
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5, 5)), Swish(),
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5)), Swish(),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)), Swish(),
                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)), Swish(),
                                  nn.MaxPool2d(kernel_size=2, stride=2)
                                  )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Swish activation function https://deep-learning-study.tistory.com/556
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = nn.Sequential()
        self.swish = Swish()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x = self.residual(x) + self.skip(x)
        x = self.swish(x)
        return x


class NaiveResidualNetwork(BaseNetwork):
    def __init__(self, features, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv_list = nn.ModuleList(
            [self.make_layer(f, n, 1 if i == 0 else 2) for i, (f, n) in enumerate(zip(features, num_blocks))])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(features[-1], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.conv_list:
            x = conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


class WrapperModel(BaseNetwork):
    def __init__(self, base_model, num_classes=10, freeze=False):
        super(WrapperModel, self).__init__()
        if freeze:
            for param in base_model.parameters():
                param.requires_grad = False

        try:
            base_model.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
        except:
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model(x)
        return x


def get_pretained_model(model_name, num_classes, freeze=False):
    model = timm.create_model(model_name, pretrained=True)
    return WrapperModel(model, num_classes, freeze)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
