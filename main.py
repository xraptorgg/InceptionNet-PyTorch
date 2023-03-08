import torch
import torch.nn as nn
import tiny_imagenet_data as dt
import model_functions as mf
import googlenet_model as goog


# hyperparameters

DATA_PATH = "inception-net/InceptionNet-PyTorch/tiny-imagenet-200"
BATCH_SIZE = 16
CLASSES = 200
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
STEP_SIZE = 8
GAMMA = 0.96
EPOCHS = 73


# device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"

# train and test metrics 

TRAIN_LOSS_LIST = []
VAL_LOSS_LIST = []
VAL_ACC_LIST = []
EPOCH_COUNT_LIST = []




if __name__ == "__main__":
    
    # create data

    train_data, val_data = dt.prepare_dataset(data_path = DATA_PATH)

    train_dataloader, val_dataloader = dt.prepare_dataloader(batch_size = BATCH_SIZE, training_data = train_data, val_data = val_data)

    # instantiating models

    googlenet = goog.GoogLeNet(CLASSES).to(device)

    # loss function and optimizer

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(params = googlenet.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    learning_decay = torch.optim.lr_scheduler.StepLR(optimizer = sgd, step_size = STEP_SIZE, gamma = GAMMA)


    # model training

    torch.manual_seed(1234)
    mf.model_train(device = device, epochs = EPOCHS, model = googlenet, train_dataloader = train_dataloader, val_dataloader = val_dataloader, 
                    loss_func = cross_entropy, optimizer = sgd, scheduler = learning_decay, epoch_count = EPOCH_COUNT_LIST, 
                    train_loss_values = TRAIN_LOSS_LIST, val_loss_values = VAL_LOSS_LIST,
                    val_acc_values = VAL_ACC_LIST)