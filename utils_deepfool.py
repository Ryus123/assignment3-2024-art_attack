import torch
import torch.nn as nn
import torch.optim as optim
from model import Net


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
            x = x + (1 + overshoot) * r_i.unsqueeze(0)
            x = x.clone().detach()
            x.requires_grad = True
            
            r_tot[i] += r_i.squeeze(0)
    
    perturbed_images = perturbed_images + (1 + overshoot) * r_tot
    return torch.clamp(perturbed_images, -1, 1)