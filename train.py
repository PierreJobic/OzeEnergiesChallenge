""" This module allows to train the different networks. 
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(EPOCHS, indices, net, dataloader, optimizer, scheduler, loss_function):
    # Predictions to use:
    BATCH_SIZE = dataloader.batch_size
    # Use of GPU ?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    index = torch.tensor(indices).to(device)

    # Prepare loss history
    print(net)
    hist_loss = np.zeros(EPOCHS)
    net.train()
    for idx_epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(dataloader.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{EPOCHS}]") as pbar:
            # Current model only use 37 Time Series to predict the 8 Time Series 
            for idx_batch, data in enumerate(dataloader):
                x, y = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # Propagate input
                output = net(x)

                # Comupte loss
                y = torch.index_select(y, 2, index)
                loss = loss_function(output, y)

                # Backpropage loss
                loss.backward()

                # Update weights
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
                pbar.update(BATCH_SIZE)
        
        # Decreasing Learning Rate
        scheduler.step()

        # Update history loss
        hist_loss[idx_epoch] = running_loss/len(dataloader)

    plt.plot(hist_loss)
    plt.show()
    print(f"Loss: {float(hist_loss[-1]):5f}")
    return hist_loss