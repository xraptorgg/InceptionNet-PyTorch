"""
Library of necessary functions required to train models on PyTorch
"""


# importing necessary libraries

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt



# training loop

def model_train(device, epochs, model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, epoch_count, train_loss_values, val_loss_values, val_acc_values):
    """ 
    Custom function to train torch models.
    Args:
        device (string): Device to train model on.
        epochs (int): Number of epochs to train the model.
        model (model instance): Model to be trained.
        train_dataloader (DataLoader): Training dataloader.
        val_dataloader (DataLoader): Validation dataloader.
        loss_func (function): Loss function.
        optimizer (optimizer): Optimizer.
        scheduler (lr_scheduler): Learning rate scheduler.
        epoch_count (List): List of epoch counts when loss were logged.
        train_loss_values (List): List of training loss values at each epoch interval.
        val_loss_values (List): List of validation loss values at each epoch interval.
        val_acc_values (List): List of Accuracy at each epoch interval.
    """

    # turn on training mode
    model.train()

    #check training device
    print(f"Training on {device}.")

    # loop through each epoch
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}\n-------------")

        # loop through each batch
        train_loss, train_acc = 0, 0
        total_steps = 1
        for images, classes in train_dataloader:

            #send data to device
            images, classes = images.to(device), classes.to(device)

            # computer forward pass
            aux1, aux2, out = model(images)

            # compute loss for main output
            loss_out = loss_func(out, classes)
            train_acc += accuracy_fn(y_true = classes, y_pred = out.argmax(dim=1))

            # compute loss for auxiliary classifiers
            loss_aux1 = loss_func(aux1, classes)
            loss_aux2 = loss_func(aux2, classes)

            # compute total loss
            loss = loss_out + 0.3 * (loss_aux1 + loss_aux2)
            train_loss += loss

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = train_loss / total_steps
            batch_acc = train_acc / total_steps

            if total_steps % 10 == 0:
                print(f"Training Loss: {batch_loss:.5f} - Training Accuracy: {batch_acc:.5f}%")

            total_steps += 1

        # learning rate decay
        scheduler.step()

        # performance on test set
        # turn on inference mode
        with torch.inference_mode():
            # loop through each batch
            total_val_loss, val_acc = 0, 0
            for val_images, val_classes in val_dataloader:
                # send data to device
                val_images, val_classes = val_images.to(device), val_classes.to(device)

                # forward pass
                y_val_pred = model(val_images)

                # compute loss
                val_loss = loss_func(y_val_pred, val_classes)
                total_val_loss += val_loss
                val_acc += accuracy_fn(y_true = val_classes, y_pred = y_val_pred.argmax(dim=1)
                )
            
            total_val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        print(f"[After {epoch + 1} epochs: Train Loss: {train_loss:.5f} - Train Accuracy: {train_acc:.5f}% - Validation Loss: {total_val_loss:.5f} - Validation Accuracy: {val_acc:.5f}%]")

        train_loss_values.append(train_loss.item())
        val_loss_values.append(total_val_loss.item())
        val_acc_values.append(val_acc)
        epoch_count.append(epoch + 1)



# test loop

def model_test(device, model, dataloader, loss_func):
    """ 
    Custom function to evaluate torch models.
    Args:
        device (string): Device to evaluate model on.
        model (model instance): Model to be evaluated.
        dataloader (DataLoader): Test dataloader.
        loss_func (function): Loss function.
    """

    # turn on test mode
    model.eval()
    
    # turn on inference mode
    with torch.inference_mode():
        # loop through each batch
        test_loss, test_acc = 0, 0
        for images, classes in dataloader:
            # send data to device
            images, classes = images.to(device), classes.to(device)

            # forward pass
            y_pred = model(images)

            # compute loss
            loss = loss_func(y_pred, classes)
            test_loss += loss
            test_acc += accuracy_fn(y_true = classes, y_pred = y_pred.argmax(dim=1)
            )
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f"Loss: {test_loss:.5f} - Accuracy: {test_acc:.5f}%")




# accuracy function

def accuracy_fn(y_true, y_pred):
    """ 
    Function to calculate accuracy.
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values
    Return:
        acc (float): Accuracy percentage
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# plotting loss curves

def plot_loss_curve(epoch_count, train_loss_values, val_loss_values):
    """
    Function to plot training loss and test loss.
    Args:
        epoch_count (List): List of epoch counts when loss were logged.
        train_loss_values (List): List of training loss values at each epoch interval.
        val_loss_values (List): List of validation loss values at each epoch interval.
    """

    plt.figure(figsize=(13, 7))
    plt.plot(epoch_count, train_loss_values, label = "Train loss")
    plt.plot(epoch_count, val_loss_values, label = "Validation loss")
    plt.title("Loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()



# plotting accuracy curve

def plot_accuracy_curve(epoch_count, val_acc_values):
    """
    Function to plot accuracy.
    Args:
        epoch_count (List): List of epoch counts when loss were logged.
        val_acc_values (List): List of Accuracy at each epoch interval.
    """

    plt.figure(figsize=(13, 7))
    plt.plot(epoch_count, val_acc_values, label = "Accuracy")
    plt.title("Accuracy curves")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()