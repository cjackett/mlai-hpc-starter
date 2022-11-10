import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from torch.nn import functional as F


def log_confusion_matrix(logger, confusion_matrix, current_epoch) -> None:
    df_cm = pd.DataFrame(confusion_matrix.numpy(), index=range(10), columns=range(10))
    plt.figure(figsize=(10, 8))
    fig = sn.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
    plt.close(fig)
    logger.add_figure("Confusion matrix", fig, current_epoch)


def log_image_examples(logger, dataset) -> None:
    figure = plt.figure(figsize=(10, 8))
    it = iter(dataset)
    for i in range(20):
        image, label = next(it)
        figure.add_subplot(4, 5, i + 1, title=label, xticks=[], yticks=[])
        image = image.cpu().clone().detach().numpy().transpose(1, 2, 0)
        plt.tight_layout()
        plt.imshow(image, cmap=plt.cm.binary)
    logger.add_figure("Training Images", figure, 0)


def log_image_classifications(logger, dataset, forward, current_epoch) -> None:
    figure = plt.figure(figsize=(10, 8))
    images, labels = next(iter(dataset))
    output = forward(images)
    _, preds = torch.max(output, 1)
    probs = F.softmax(output, dim=1)
    for i in range(20):
        ax = figure.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.set_title("{} ({:.0f}%)\n{}".format(preds[i].item(), probs[i][preds[i]]*100, labels[i].item()), color=("green" if preds[i] == labels[i] else "red"))
        image = images[i].detach().numpy().transpose(1, 2, 0)
        plt.tight_layout()
        plt.imshow(image, cmap=plt.cm.binary)
    logger.add_figure("Model Predictions", figure, current_epoch)


def log_projector(logger, dataset, n=100) -> None:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    images, labels = next(iter(dataloader))
    features = images.view(-1, 28 * 28)
    logger.add_embedding(features, metadata=labels, label_img=images)
