import torch
import torch.nn as nn
from torch import optim
from cosine_annealing_warmup import \
    CosineAnnealingWarmupRestarts  # https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
from dataset import *
from models import *
import timm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True


def save_figure(report_dict, num_epochs, file_name):
    fig = plt.figure()
    epochs = list(range(num_epochs))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax1.plot(epochs, report_dict['train_loss'])
    ax1.plot(epochs, report_dict['valid_loss'])

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    ax2.plot(epochs, report_dict['train_acc'])
    ax2.plot(epochs, report_dict['valid_acc'])

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.plot(epochs, report_dict['lr'])

    plt.savefig(file_name)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = {
        'batch_size': 512,
        'base_lr': 1e-3,
        'num_epochs': 30,
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

    num_classes = 10
    model_name = 'residual_rotation'
    # model = NaiveNetwork(num_classes)
    model = NaiveResidualNetwork(features=[64, 128, 256], num_blocks=[2, 2, 2], num_classes=num_classes)
    # model = get_pretained_model(model_name,num_classes=num_classes, freeze=False)
    # model_name += '_rotation_nf'

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=num_epochs,
                                              cycle_mult=1.0,
                                              max_lr=base_lr,
                                              min_lr=1e-8,
                                              warmup_steps=0,
                                              gamma=0.5)
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.5)
    model.to(device)
    criterion.to(device)

    # model.load_state_dict(torch.load(f'best_model_acc_{model_name}.pth'))
    # test_loss, test_acc = model.valid_process(criterion, testloader, device, verbose=False)
    # print(f'Test Loss: {test_loss} Acc: {test_acc}')
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    test_loss, test_acc = model.valid_process(criterion, testloader, device, verbose=False)
    print(f'Test Loss: {test_loss} Acc: {test_acc}')
    exit()

    # optimizer.param_groups[0]['lr']=base_lr
    # rotation_trainloader = get_rotation_dataloader(batch_size=batch_size * 4)
    # for epoch in range(num_epochs):
    #     train_loss, acc = model.train_process(optimizer, criterion, rotation_trainloader, device, verbose=False)
    #     print(f'Epoch {epoch} Loss: {train_loss} Acc: {acc}')
    #
    # model.fc = nn.Sequential(nn.Linear(256, 128),
    #                          Swish(),
    #                          nn.Linear(128, 10))
    #
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    # scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                           first_cycle_steps=num_epochs,
    #                                           cycle_mult=1.0,
    #                                           max_lr=base_lr,
    #                                           min_lr=1e-8,
    #                                           warmup_steps=0,
    #                                           gamma=0.5)
    # criterion = LabelSmoothingLoss(10, smoothing=0.5)
    # model.to(device)
    # criterion.to(device)

    report_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [], 'valid_loss': [], 'lr': []}
    optimizer.param_groups[0]['lr']=base_lr
    best_loss = float('inf')
    best_acc = 0
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
            best_loss = valid_loss
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')
            print(f'Best Model is saved. {epoch} - {best_loss}')
        if best_acc < valid_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'best_model_acc_{model_name}.pth')
            print(f'Best Acc Model is saved. {epoch} - {best_acc}')

    save_figure(report_dict, num_epochs, file_name=f'./pictures/result_{model_name}.png')

    model.load_state_dict(torch.load(f'best_model_acc_{model_name}.pth'))
    test_loss, test_acc = model.valid_process(criterion, testloader, device, verbose=False)
    print(f'Test Loss: {test_loss} Acc: {test_acc}')

    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    test_loss, test_acc = model.valid_process(criterion, testloader, device, verbose=False)
    print(f'Test Loss: {test_loss} Acc: {test_acc}')
