#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 

def deepfool_attack(model, images, labels, device, num_classes=10, overshoot=0.02, max_iter=50):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    perturbed_images = images.clone().detach()
    
    batch_size = images.size(0)
    r_tot = torch.zeros_like(perturbed_images).to(device)
    
    for i in range(batch_size):
        x = perturbed_images[i].unsqueeze(0)
        x.requires_grad = True
        original_label = labels[i].item()

        for _ in range(max_iter):
            x.grad = None
            outputs = model(x)
            _, pred = outputs.max(1)
            
            if pred.item() != original_label:
                break
            
            # Compute gradients and linearized decision boundaries
            grad_original = torch.autograd.grad(outputs[0, original_label], x, retain_graph=True)[0]
            
            # Find minimal perturbation
            min_perturb = float('inf')
            best_perturb = None
            
            for k in range(num_classes):
                if k == original_label:
                    continue
                
                grad_k = torch.autograd.grad(outputs[0, k], x, retain_graph=True)[0]
                
                # Linearized decision boundary difference
                w_k = grad_k - grad_original
                f_k = outputs[0, k] - outputs[0, original_label]
                
                # Project perturbation
                perturb = abs(f_k) / (torch.norm(w_k.flatten()) + 1e-8)
                
                if perturb < min_perturb:
                    min_perturb = perturb
                    best_perturb = w_k
            
            # Compute minimal perturbation
            r_i = min_perturb * best_perturb / (torch.norm(best_perturb.flatten()) + 1e-8)
            
            # Overshoot
            # x = x + (1 + overshoot) * r_i.unsqueeze(0)
            x = x + (1 + overshoot) * r_i
            x = x.clone().detach()
            x.requires_grad = True
            
            r_tot[i] += r_i.squeeze(0)
    
    perturbed_images = perturbed_images + (1 + overshoot) * r_tot
    return torch.clamp(perturbed_images, -1, 1)

def deepfool_train_adversarial_model(net, train_loader, pth_filename, num_epochs, device, train_on_both=True):
    """
    Enhanced adversarial training function using DeepFool attack
    
    Args:
    - net: Neural network model
    - train_loader: DataLoader for training data
    - pth_filename: Path to save the trained model
    - num_epochs: Number of training epochs
    - device: Computing device (cuda/cpu)
    - train_on_both: Whether to train on both clean and adversarial examples
    """
    print("Starting adversarial training")
    
    # Use NLLLoss consistent with the second snippet
    criterion = nn.NLLLoss()
    
    # SGD optimizer with momentum, matching the second snippet
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Generate adversarial examples using DeepFool
            images_adv = deepfool_attack(net, images, labels, device)
            
            if train_on_both:
                # Combine clean and adversarial examples
                images_combined = torch.cat((images, images_adv), 0)
                labels_combined = torch.cat((labels, labels), 0)
            else:
                # Train only on adversarial examples
                images_combined = images_adv
                labels_combined = labels
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = net(images_combined) # Forward pass
            
            loss = criterion(outputs, labels_combined) # Compute loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 500 == 499:  # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    
    # Save the trained model
    torch.save(net.state_dict(), pth_filename)
    print(f'Model saved in {pth_filename}')

#  unnormalize and convert tensor to numpy array for visualization
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def visualize_images(images, labels, preds, title):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i in range(len(images)):
        axs[i].imshow(np.transpose(images[i], (1, 2, 0)))
        axs[i].set_title(f'{title}\nTrue: {classes[labels[i]]}\nPred: {classes[preds[i]]}')
        axs[i].axis('off')
    plt.show(block=False)

def test_deepfool(model, testloader, device):
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Get the original prediction
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Apply DeepFool attack
        adv_images = deepfool_attack(model, images, labels, device)

        # Get the perturbed prediction
        outputs_perturbed = model(adv_images)
        _, preds_perturbed = torch.max(outputs_perturbed, 1)

        correct += (preds_perturbed == labels).sum().item()
        total += labels.size(0)

        # if batch_idx == 0:
        #     images_to_show = images[:5].cpu()
        #     adv_to_show = adv_images[:5].detach().cpu()
        #     labels_to_show = labels[:5].cpu().numpy()
        #     preds_to_show = preds[:5].cpu().numpy()
        #     preds_perturbed_to_show = preds_perturbed[:5].cpu().numpy()

        #     visualize_images(images_to_show, labels_to_show, preds_to_show, title="Original Images")
        #     visualize_images(adv_to_show, labels_to_show, preds_perturbed_to_show, title="Adversarial Images")

    acc = 100 * correct / total
    return acc

