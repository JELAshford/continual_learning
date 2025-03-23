from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch

import matplotlib.pylab as plt
from tqdm import tqdm

torch.manual_seed(1701)


class ClassSubsetDataset(Dataset):
    def __init__(self, full_dataset, class_labels):
        class_mask = torch.isin(full_dataset.targets, torch.Tensor(class_labels))
        self.class_samples = full_dataset.data[class_mask, :].float()
        self.class_labels = full_dataset.targets[class_mask]

    def __len__(self):
        return len(self.class_samples)

    def __getitem__(self, idx):
        return self.class_samples[idx : (idx + 1), :], self.class_labels[idx]


def _make_infinite(dloader):
    while True:
        yield from dloader


def test(model, device, test_loader, num_targets=10, eps=1e-8):
    """Test the model on a dataloader"""
    model.eval()
    # Accumulate data and predictions from each batch
    preds, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            targets.append(target)
            preds.append(output.argmax(dim=1))
    preds, targets = torch.cat(preds), torch.cat(targets)

    # Calculate per-class accuracy
    class_counts = torch.bincount(targets, minlength=10)
    correct_counts = torch.bincount(targets[preds == targets], minlength=10)
    class_accuracy = correct_counts / (class_counts + eps)
    return class_accuracy


if __name__ == "__main__":
    # Training settings
    BATCH_SIZE = 64
    TRAIN_STEPS = 1000
    LOG_STEPS = 50
    LEARNING_RATE = 3e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    transform = transforms.Compose(transform_list)
    mnist_options = dict(root="./rsc/data", download=True, transform=transform)

    train_dataset = datasets.MNIST(**mnist_options, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dataloaders = [
        DataLoader(
            ClassSubsetDataset(train_dataset, class_labels=pair),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        for pair in zip(range(0, 10, 2), range(1, 11, 2))
    ]

    test_dataset = datasets.MNIST(**mnist_options, train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize the model, optimizer
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(7 * 7 * 32, 64),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1),
    )
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # # Quick warmup train on the full dataset
    # warmup_iterator = enumerate(_make_infinite(train_dataloader))
    # for step_idx, (data, target) in (
    #     pb := tqdm(warmup_iterator, total=TRAIN_STEPS // 2)
    # ):
    #     if step_idx > TRAIN_STEPS // 2:
    #         break
    #     data, target = data.to(DEVICE), target.to(DEVICE)
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = F.nll_loss(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     pb.set_description(f"warmup_loss={torch.mean(loss):.4f}")

    # Train and selected datasets and evaluate
    total_batches = 0
    log_times, log_accuracies = [], []
    for dataloader in dataloaders:
        model.train()
        data_iterator = enumerate(_make_infinite(dataloader))
        for step_idx, (data, target) in (pb := tqdm(data_iterator, total=TRAIN_STEPS)):
            if step_idx > TRAIN_STEPS:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_batches += 1
            if total_batches % LOG_STEPS == 0:
                class_accuracies = test(model, DEVICE, test_loader)
                pb.set_description(f"overall_acc={torch.mean(class_accuracies):.4f}")
                log_times.append(total_batches)
                log_accuracies.append(class_accuracies)

    # Visualise accuracy over training
    log_accuracies = torch.stack(log_accuracies)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(log_times, log_accuracies)
    fig.legend(labels=[*map(str, range(10))], loc="right")
    ax.set_xlabel("Training Batches")
    ax.set_ylabel("Test Dataset Accuracy")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(
        log_accuracies.T, vmin=0, vmax=1, aspect="auto", interpolation="none"
    )
    ax.set_yticks(ticks=range(10), labels=[*map(str, range(10))])
    fig.colorbar(im)
    ax.set_xlabel("Training Batches")
    ax.set_ylabel("MNIST Digit Class")
    plt.show()
