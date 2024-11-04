import os
current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
target_folder_path = os.path.join(parent_parent_dir, "dataset_generation")
import pandas as pd 
import numpy as np
from decoder import Decoder
import embedding 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import utils
import loss
import models
from tqdm import tqdm

samples_num = 300
qubits_num = 8
shots_num = 64
test_size = 0.8

def main(qubits_num, train_samples, test_samples):
    utils.fix_seed(2024)
    
    try:
        finetune_path = "/heisenberg_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        df = pd.read_csv(target_folder_path + finetune_path)
    except:
        raise FileNotFoundError("Dataset not found")
    
    embedding_dim = 512
    hidden_dim = 128
    seq_len = qubits_num + 1
    batch_size = samples_num
    
    meas_records = np.array([utils.read_matrix_v2(x) for x in df['measurement_samples'].values]).reshape(-1, qubits_num, shots_num)
    conditions = np.array([utils.read_matrix_v2(x) for x in df['coupling_matrix'].values])
    
    all_idx = np.random.choice(range(samples_num), batch_size, replace=False)
    batch_conditions = conditions[all_idx]
    batch_measures = meas_records[all_idx]
    cls_token = torch.zeros((batch_size, shots_num, 1), dtype=torch.long)
    batch_measures = torch.cat((cls_token, torch.tensor(batch_measures).permute(0, 2, 1).long()), dim=2).permute(0, 2, 1).float()
    
    y_approx_corr = torch.tensor([utils.read_matrix_v2(x) for x in df['approx_correlation_matrix'].values])
    y_exact_corr = torch.tensor([utils.read_matrix_v2(x) for x in df['exact_correlation_matrix_zz'].values])
    
    rnn = nn.LSTM(shots_num, embedding_dim, 1)
    token_embedding_ft, _ = rnn(batch_measures)
    all_embedding = token_embedding_ft + embedding.get_embedding_ft(batch_size, seq_len, embedding_dim, batch_conditions)
    
    #test_samples = int(samples_num * test_size)
    #train_samples = samples_num - test_samples
    
    train_sample_idx = np.random.choice(range(100), train_samples, replace=False)
    # test_sample_idx = np.array([i for i in range(samples_num) if i not in train_sample_idx])
    test_sample_idx = np.arange(100, 300, 1)
    X_train = all_embedding[train_sample_idx]
    y_train = y_approx_corr[train_sample_idx]
    X_test = all_embedding[test_sample_idx]
    y_test = y_exact_corr[test_sample_idx]
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test) 
    
    
    decoder = Decoder(embedding_dim, seq_len, embedding_dim, ffn_hidden=128, n_head=8, n_layers=qubits_num, drop_prob=0.1, device='cuda')
    try:
        decoder.load_state_dict(torch.load("save/pretrain_q{}_s1024_bs100_ep1000.pt".format(qubits_num), weights_only=True))
        print("Pretrained model loaded.")
    except:
        raise FileNotFoundError("Pretrained model not found.")
    
    # supervised fine-tuning
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    
    finetune_model = models.FinetuneDecoder(decoder, None, embedding_dim, hidden_dim, embedding_dim, qubits_num*qubits_num)
    epochs = 2000
    for i in tqdm(range(epochs)):
        total_loss = 0.0
        for input, target in DataLoader(train_dataset, batch_size=len(train_dataset)):
            train_loss = 0.0
            optimizer.zero_grad()
            output = finetune_model(input)
            train_loss = loss.rmse_loss(output, target)
            gradients = torch.autograd.grad(train_loss, finetune_model.parameters(), retain_graph=True)
            for param, grad in zip(finetune_model.parameters(), gradients):
                param.grad = grad
            optimizer.step()
            total_loss += train_loss.item()
        if(i % 100 == 0):
            print("Epoch: {}, Loss: {}".format(i, total_loss / len(train_dataset)))

    result = 0.0
    # evaluation
    finetune_model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for input, target in DataLoader(test_dataset, batch_size=len(test_dataset)):
            output = finetune_model(input)
            test_loss = loss.rmse_loss(output, target)
        print("Test Loss: {}".format(test_loss.item()))
        result = test_loss.item()
    return result

if __name__ == "__main__":
    qubits_list = [8, 10, 12]
    train_samples_list = [20, 50, 90]
    test_samples = 200
    for qubits_num in qubits_list:
        for train_samples in train_samples_list:
            tloss = main(qubits_num, train_samples, test_samples)
            with open("results/heisenberg_1d_correlation_rmse.txt", "a") as f:
                f.write("qubits: {}, train_samples: {}, test loss: {}\n".format(qubits_num, train_samples, tloss))
                f.close()
        
    

