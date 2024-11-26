import torch
import torch.nn as nn
import torch.optim as optim



def fgsm_attack(model, loss_fn, images, labels, epsilon, device):

    images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = loss_fn(outputs, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Gradients
    grad = images.grad.data

    # Generate perturbed image 
    perturbed_images = images + epsilon * grad.sign()

    # Clamp the perturbed images to [-1,1] since we normalized the images
    perturbed_images = torch.clamp(perturbed_images, -1, 1)

    return perturbed_images


def pgd_attack(model, loss_fn, images, labels, epsilon, alpha, num_iter, device):

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()

    for _ in range(num_iter):
        images.requires_grad = True

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Gradients
        grad = images.grad.data

        # Update the images
        adv_images = images + alpha * grad.sign()

        # Clamp perturbation
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=-1, max=1).detach()

    return images


def train_adversarial_model(net, train_loader, pth_filename, num_epochs, device, train_on_both=True):
    '''Adversarial training function'''
    print("Starting adversarial training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
   
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Generate adversarial examples using the specified attack method
            images_adv = pgd_attack(net, criterion, images, labels, epsilon=0.1, alpha=0.01, num_iter=5, device=device)

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

            # Forward pass
            outputs = net(images_combined)
            loss = criterion(outputs, labels_combined)

            # Backward and optimize
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Save the trained model
    torch.save(net.state_dict(), pth_filename)
    print('Model saved in {}'.format(pth_filename))