import torch
import torch.nn as nn
from torch import optim
from cosine_annealing_warmup import \
    CosineAnnealingWarmupRestarts  # https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
from dataset import *
from models import *

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = {
        'batch_size': 128,
        'base_lr': 1e-3,
        'num_epochs': 500,
        'valid_ratio': 0.1,
        'seed': 1213,
        'device': device
    }
    batch_size = config['batch_size']
    base_lr = config['base_lr']
    num_epochs = config['num_epochs']
    valid_ratio = config['valid_ratio']
    seed = config['seed']
    print(config)

    trainloader, validloader, testloader = get_dataloader(batch_size=batch_size, valid_ratio=valid_ratio, seed=seed)

    model = NaiveNetwork()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=20,
                                              cycle_mult=1.0,
                                              max_lr=1e-3,
                                              min_lr=1e-8,
                                              warmup_steps=0,
                                              gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    report_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [], 'valid_loss': [], 'lr': []}

    best_loss = float('inf')
    best_state = None
    for epoch in range(num_epochs):
        train_loss, acc = model.train_process(optimizer, criterion, trainloader, device, verbose=False)
        valid_loss, valid_acc = model.valid_process(criterion, validloader, device, verbose=False)

        lr = optimizer.param_groups[0]['lr']
        print(
            f'\rEpoch {epoch} Total Loss: {train_loss}, Acc: {acc} - Valid Loss: {valid_loss} Acc:{valid_acc} lr: {lr}')
        scheduler.step()

        report_dict['train_acc'].append(acc)
        report_dict['train_loss'].append(train_loss)
        report_dict['valid_acc'].append(valid_acc)
        report_dict['valid_loss'].append(valid_loss)
        report_dict['lr'].append(lr)

        if best_loss > valid_loss:
            best_state = model.state_dict()
            best_loss = valid_loss
            torch.save(model.state_dict(), f'best_model.pth')
            print(f'Best Model is saved. {epoch} - {best_loss}')

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = model.valid_process(criterion, testloader, device, verbose=False)
    print(f'Test Loss: {test_loss} Acc: {test_acc}')
