from dataset import *
from model import *
from train import train
from save import save
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

def main(net, method, LR=1e-2, EPOCHS=15, dataloader=None, add_time=False, normalize=True, indices=list(range(8)), save_result=False):
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # DataLoader
    if dataloader == None:
        dataloader = DataLoader(OzeDataset(dataset_x_path="../data/x_train.csv", 
                                            dataset_y_path="../data/y_train.csv",
                                            labels_path="../data/labels.json",
                                            normalize=normalize,
                                            add_time=add_time),
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)
      
    m, M = dataloader.dataset.m, dataloader.dataset.M

    #net = TestRecurrentNet('lstm', input_dim=38, batch_first=True)

    # Model
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=10, 
                                                gamma=0.1, 
                                                last_epoch=-1)
    loss_function = nn.MSELoss()

    # Train
    train(EPOCHS, indices, net, dataloader, optimizer, scheduler, loss_function)

    # Save
    if save_result:
        dataset_eval = OzeEvaluationDataset(dataset_x_path="../data/x_test.csv", 
                                            labels_path="../data/labels.json", 
                                            add_time=True)
        save(dataset_eval, net, m, M, method, denormalize=normalize)


if __name__ == '__main__':
    net = BenchmarkLSTM(input_dim = 37)
    method = 'BenchmarkLSTM'
    main(net, method, save_result=False)