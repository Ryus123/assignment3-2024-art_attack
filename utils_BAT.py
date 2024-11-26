#!/usr/bin/env python3 
"""
Compute tools for the Boosted Adversarial Training Algorithm
https://proceedings.mlr.press/v119/pinot20a/pinot20a.pdf
"""

import torch
import torch.nn as nn
import torch.optim as optim

from utils_ADV import pgd_attack


def train_boosted_adversarial_model(h1, h2, train_loader, pth_filename, num_epochs, device):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(h2.parameters(), lr=0.001, momentum=0.9)
    print("Starting Boosted Adversarial Training")

    for epoch in range(num_epochs):
        h2.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            adversarial_data = pgd_attack(h1, criterion, images, labels, epsilon=0.1, alpha=0.01, num_iter=5, device=device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = h2(adversarial_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    h2.save(pth_filename)
    print('BAT Model saved in {}'.format(pth_filename))
    

def test_BAT(h1, h2, test_loader, device, alpha=0.2):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate e return the mixture constructed with those two classifiers
            outputs_h1 = h1(images)
            outputs_h2 = h2(images)
            outputs = (1-alpha)*outputs_h1 + alpha*outputs_h2
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total