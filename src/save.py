""" This module allows to save the output prediction of the network. 
"""

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

def save(dataset_eval, net, m, M, method, denormalize=True):
    # m_test = 500
    m_test = len(dataset_eval)
    K = 672

    # Make Predictions:
    predictions = np.zeros((m_test, K, 8))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(device)
    net.eval()
    with torch.no_grad():
        for idx, line in tqdm(enumerate(dataset_eval), total=m_test):
            # Run prediction
            netout = net(torch.Tensor(line[np.newaxis, :, :]).to(device)).cpu().numpy()
            # De-normalize output
            if denormalize == True:
              output = netout * (M - m + np.finfo(float).eps) + m
            else:
              output = netout
            predictions[idx] = output

    # Transform predictions to DataFrame
    lines_output = np.transpose(predictions, axes=(0, 2, 1))
    lines_output[:, 1, :] = np.zeros((m_test,K))
    lines_output = lines_output.reshape(m_test, -1)
    csv_header = [f"{var_name}_{k}" for var_name in dataset_eval.labels['X'] for k in range(K)]

    dataframe_index = pd.DataFrame(np.arange(7500, 7500+m_test).reshape(-1, 1), columns=['index'])
    dataframe_output = pd.DataFrame(lines_output, columns=csv_header)
    data = data = pd.concat([dataframe_index, dataframe_output], axis=1)

    # Export as CSV
    data.to_csv('../data/y_out/y_' + str(method) + '.csv', index=False)