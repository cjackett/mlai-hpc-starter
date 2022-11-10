import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from torch.nn import functional as F


def log_confusion_matrix(logger, confusion_matrix, current_epoch) -> None:
    df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(10), columns=range(10))
    plt.figure(figsize=(10, 8))
    fig = sn.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
    plt.close(fig)
    logger.add_figure("Confusion matrix", fig, current_epoch)


def log_image_examples(logger, dataset) -> None:
    figure = plt.figure(figsize=(10, 8))
    images, labels = next(iter(dataset))
    for i in range(20):
        figure.add_subplot(4, 5, i + 1, title=labels[i].item(), xticks=[], yticks=[])
        image = images[i].cpu().clone().detach().numpy().transpose(1, 2, 0)
        image = image * np.array((0.2470, 0.2435, 0.2616)) + np.array((0.4914, 0.4822, 0.4465))
        image = image.clip(0, 1)
        plt.tight_layout()
        plt.imshow(image, cmap=plt.cm.binary)
    logger.add_figure("Training Images", figure, 0)


def log_image_classifications(logger, dataset, forward, current_epoch) -> None:
    figure = plt.figure(figsize=(10, 8))
    images, labels = next(iter(dataset))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = forward(images.to(device))
    _, preds = torch.max(output, 1)
    probs = F.softmax(output, dim=1)
    for i in range(20):
        ax = figure.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.set_title("{} ({:.1f}%)\n{}".format(preds[i].item(), probs[i][preds[i]]*100, labels[i].item()), color=("green" if preds[i] == labels[i] else "red"))
        image = images[i].cpu().clone().detach().numpy().transpose(1, 2, 0)
        image = image * np.array((0.2470, 0.2435, 0.2616)) + np.array((0.4914, 0.4822, 0.4465))
        image = image.clip(0, 1)
        plt.tight_layout()
        plt.imshow(image, cmap=plt.cm.binary)
    logger.add_figure("Model Predictions", figure, current_epoch)


def log_projector(logger, dataset, n=100) -> None:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    images, labels = next(iter(dataloader))
    features = images.view(-1, 28 * 28)
    logger.add_embedding(features, metadata=labels, label_img=images)
