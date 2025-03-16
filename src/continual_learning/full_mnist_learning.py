from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch

import matplotlib.pylab as plt
from tqdm import tqdm


# Define the CNN architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class SingleClassDatset(Dataset):
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
    BATCH_SIZE = 32
    TRAIN_STEPS = 1000
    LOG_STEPS = 100
    LEARNING_RATE = 3e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    transform = transforms.Compose(transform_list)
    mnist_options = dict(root="./rsc/data", download=True, transform=transform)

    train_dataset = datasets.MNIST(**mnist_options, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # dataloaders = [
    #     DataLoader(
    #         SingleClassDatset(train_dataset, class_labels=[label]),
    #         batch_size=BATCH_SIZE,
    #         shuffle=True,
    #     )
    #     for label in range(3)
    # ]
    # dataloaders = [
    #     DataLoader(
    #         SingleClassDatset(
    #             train_dataset, class_labels=[i for i in range(10) if i % 2 == 0]
    #         ),
    #         batch_size=BATCH_SIZE,
    #         shuffle=True,
    #     ),
    #     DataLoader(
    #         SingleClassDatset(
    #             train_dataset, class_labels=[i for i in range(10) if i % 2 != 0]
    #         ),
    #         batch_size=BATCH_SIZE,
    #         shuffle=True,
    #     ),
    # ]
    dataloaders = [
        DataLoader(
            SingleClassDatset(train_dataset, class_labels=pair),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        for pair in zip(range(0, 10, 2), range(1, 11, 2))
    ]

    test_dataset = datasets.MNIST(**mnist_options, train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize the model, optimizer
    model = MNISTNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train and evaluate
    total_batches = 0
    log_times, log_accuracies = [], []
    for label, dataloader in enumerate(dataloaders):
        model.train()
        data_iterator = enumerate(_make_infinite(dataloader))
        for step_idx, (data, target) in tqdm(data_iterator, total=TRAIN_STEPS):
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
                # print(f"{label=} {total_batches=} {class_accuracies=}")
                log_times.append(total_batches)
                log_accuracies.append(class_accuracies)

    # Visualise accuracy over training
    log_accuracies = torch.stack(log_accuracies)
    plt.plot(log_times, log_accuracies)
    plt.legend(labels=[*map(str, range(10))])
    plt.show()
    plt.imshow(log_accuracies)
    plt.show()
