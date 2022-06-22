'''
Author: HelinXu xuhelin1911@gmail.com
Date: 2022-06-15 21:59:33
LastEditTime: 2022-06-23 03:41:04
Description: Training code based on DenseNet
'''

import torch
import time
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import random
import argparse

from dataset import TaobaoDataset
from misc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# tune the model
def train_model(model, dataloaders, criterion, optimizer, writer, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        for i, (inputs, labels) in tqdm(enumerate(dataloaders['train'])):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = model(inputs)

            loss = criterion(output, labels) # CrossEntropyLoss() expects the output to be log-probabilities.
            
            # backward
            loss.backward()
            
            # update
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()

            # save to tensorboard
            writer.add_scalar('loss/training', loss.item(), epoch * len(dataloaders['train']) + i)

        # print epoch statistics
        epoch_loss = running_loss / len(dataloaders['train'])
        print('Epoch {} Loss (training): {:.4f}'.format(epoch, epoch_loss))

        # save model
        torch.save(model.state_dict(), './model/densenet161_epoch_{}.pth'.format(epoch))
        # save the latest model name to txt file
        with open('./model/last_model.txt', 'w') as f:
            f.write('densenet161_epoch_{}.pth'.format(epoch))

        # validate
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0.1
            total = 0.1
            for inputs, labels in dataloaders['train']:
                # randomly pass 90% to save time.
                if random.random() > 0.1:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                loss = criterion(output, labels)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders['train'])
            epoch_acc = correct / total
            print('Epoch {} Loss (train): {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
            writer.add_scalar('loss/train', epoch_loss, epoch)
            writer.add_scalar('accuracy/train', epoch_acc, epoch)
            
            running_loss = 0.0
            correct = 0.1
            total = 0.1
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                loss = criterion(output, labels)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders['val'])
            epoch_acc = correct / total
            print('Epoch {} Loss (val): {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
            writer.add_scalar('loss/val', epoch_loss, epoch)
            writer.add_scalar('accuracy/val', epoch_acc, epoch)
            


def main(num_classes=len(LABEL_1)+1, tune_full_model=False, batch_size=32, num_epochs=25):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)

    if not tune_full_model:
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Linear(2208, num_classes)

    # Send the model to GPU
    model = model.to(device)

    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    train_loader = torch.utils.data.DataLoader(
        dataset=TaobaoDataset(train=True, data_root=DATA_ROOT),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=TaobaoDataset(train=False, data_root=DATA_ROOT),
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=True
    )


    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter('./logs')
    train_model(model, dataloaders, criterion, optimizer, writer, num_epochs=25)


if __name__ == '__main__':
    # pass args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=len(LABEL_1)+1)
    parser.add_argument('--tune_full_model', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    main(args.num_classes, args.tune_full_model, args.batch_size)