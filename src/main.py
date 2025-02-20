import copy
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
import torch.nn.functional as F
import torch.nn.utils.prune as tprune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split


class NNMnist(nn.Module):

    def __init__(self, h1=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def post_prune_model(amount):
    # Pruning
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.l1_unstructured(module, name='weight', amount=amount)

    #Remove the pruning reparametrizations to make the model explicitly sparse
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.remove(module, 'weight')


def get_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    return train_dataset, test_dataset


def train():
    lr = 0.001
    weight_decay=1e-4
    epochs = 5
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        batch_losses = []
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.enable_grad():
                model.train()
                outputs = model(images)
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
        mean_epoch_loss = sum(batch_losses) / len(batch_losses)
        losses.append(mean_epoch_loss)
    return sum(losses) / len(losses)


def test(dataset, seed, pruned=False):
    criterion = nn.NLLLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    inference_time = pd.DataFrame(columns=['Batch','Time'])
    for batch_index, (images, labels) in enumerate(data_loader):
        start_time = time.time()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        end_time = time.time()
        inference_time = inference_time._append(
            {'Batch': batch_index,'Time': end_time - start_time},
            ignore_index=True
        )
    inference_time.to_csv(f'data/inference-time-seeed-{seed}-pruned-{pruned}.csv', ignore_index=True)
    accuracy = correct / total
    return loss, accuracy

if __name__ == '__main__':

    train_dataset, test_dataset = get_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    batch_size = 32
    max_seed = 10
    
    data_output_directory = Path('data')
    data_output_directory.mkdir(parents=True, exist_ok=True)

    for seed in range(max_seed):
        print(f'Seed --- {seed}')
        model = NNMnist()
        model.to(device)
        train()
        test(test_dataset, seed)
        post_prune_model(0.5)
        test(test_dataset, seed, True)
