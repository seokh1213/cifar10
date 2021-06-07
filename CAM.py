import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from dataset import *
from models import *

plt.rcParams["figure.figsize"] = (15, 5)


def save_image(image1, image2, image3, file_name, title):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.set_xlabel('Original')
    ax1.imshow(image1)

    ax2.set_xlabel('Model')
    ax2.imshow(image2)

    ax3.set_xlabel('Self-Supervised Model')
    ax3.imshow(image3)

    plt.suptitle(title)

    plt.savefig(file_name)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=get_transform(test=True))
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # model = NaiveNetwork(num_classes=10)
    # model2 = NaiveNetwork(num_classes=10)
    model = NaiveResidualNetwork(features=[64, 128, 256], num_blocks=[2, 2, 2], num_classes=10)
    model2 = NaiveResidualNetwork(features=[64, 128, 256], num_blocks=[2, 2, 2], num_classes=10)
    # model = get_pretained_model('resnet18', num_classes=10)
    # model2 = get_pretained_model('resnet18', num_classes=10)
    # model = get_pretained_model('resnet50', num_classes=10)
    # model2 = get_pretained_model('resnet50', num_classes=10)

    model_name = 'residual'

    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    model = model.to(device)

    model2.load_state_dict(torch.load(f'best_model_{model_name}_rotation.pth'))
    model2 = model2.to(device)

    classes = testset.classes
    modules = list(model.modules())
    modules2 = list(model2.modules())

    for k in range(-1, -40, -1):
        success = True
        try:
            cam = GradCAM(model=model, target_layer=modules[k], use_cuda=device != 'cpu')
            cam2 = GradCAM(model=model2, target_layer=modules2[k], use_cuda=device != 'cpu')

            skip = 10
            cnt = 0
            for i, data in enumerate(testloader):
                if i < skip:
                    continue
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                cnt += 1
                original_image = (testset.data[i] / 255).astype(np.float32)

                result = cam(input_tensor=images, target_category=labels.item())
                result = result[0, :]
                visualization = show_cam_on_image(original_image, result)

                result = cam2(input_tensor=images, target_category=labels.item())
                result = result[0, :]
                visualization2 = show_cam_on_image(original_image, result)

                save_image(original_image, visualization, visualization2,
                           file_name=f'./pictures/CAM/{model_name}_{classes[labels.item()]}_{abs(k)}.png',
                           title=classes[labels.item()])
                if cnt == 4:
                    break
        except Exception:
            success = False
        if success:
            print(k, 'success')
