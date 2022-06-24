import torch
import torchvision
import torch.optim as optim

import argparse
import os
import sys
import numpy as np
from loguru import logger
import pandas as pd
import time
from tqdm import tqdm

from pd_data import load_data
from ShuffleNet.model import ShuffleNet
    

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='SSDH_PyTorch')
    parser.add_argument('-d', '--csv_path',
                        help='Dataset csv path.')
    parser.add_argument('-r', '--root',
                        help='Path of dataset')
    parser.add_argument('-T', '--max_iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('-w', '--num_workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')

    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='Batch size.(default: 64)')

    parser.add_argument('-C', '--checkpoint', default=None, type=str,
                        help='Path of checkpoint.')

    args = parser.parse_args()

    return args


def train_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    correct_cnt = 0
    total_cnt = 0
    
    for img_before, img_after, _ , target in tqdm(dataloader, leave=False):
        

        data = torch.cat((img_before,img_after),1).to(device)
        target = target.to(device)
        
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)
        correct_cnt += (output.argmax(dim=-1) == target).sum().item()
        total_cnt += target.size(0)

    epoch_loss = running_loss / total_cnt
    epoch_acc = correct_cnt / total_cnt
    print(f'Trainning loss = {epoch_loss:.4f}, accuracy = {epoch_acc:.4f}.')

def test(model, dataloader, criterion, device='cpu'):
    model.eval()
    test_loss = 0
    test_correct = 0
    total_num = 0
    for img_before, img_after, _ , target in tqdm(dataloader, leave=False):
        data = torch.cat((img_before,img_after),1).to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)
        test_loss+=loss.item()*target.size(0)
        test_correct+=(output.argmax(dim=-1) == target).sum().item()
        total_num +=target.size(0)
    loss = 1.*test_loss/total_num
    acc = 1.*test_correct/total_num
    print(f"Test loss = {loss:.4f}, accuracy = {acc:.4f}")

if __name__ == '__main__':

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('Running on the device:', device)

    args = load_config()

    model = ShuffleNet(num_classes=2, in_channels=6)
    model = model.to(device)

    train_dataloader, test_dataloader = load_data(args.batch_size, args.num_workers, args.root, args.csv_path)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #scheduler = ExponentialLR(optimizer, gamma=0.9)

    for i in range(args.max_iter):
        print("training on {} iters". format(i))
        train_epoch(model, criterion, optimizer, train_dataloader, device)
    
    test(model, test_dataloader, criterion, device)

    
