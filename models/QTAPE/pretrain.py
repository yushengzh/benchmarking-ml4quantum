import os
current_dir = os.getcwd()
parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
target_folder_path = os.path.join(parent_parent_dir, "dataset_generation")

import utils
import pandas as pd 
import numpy as np
from decoder import Decoder
import embedding 
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
samples_num = 100
# qubits_num = 8 # L
shots_num = 1024 # K
batch_size = 100


def main(qubits_num):
    utils.fix_seed(2024)
    try:
        dataset_path = "/heisenberg_1d/n{samples_num}|X(coupling, meas{shots})_y(energy,entropy,corrs)_q{q}.csv".format(samples_num=samples_num, shots=shots_num, q=qubits_num)
        df = pd.read_csv(target_folder_path + dataset_path)
    except:
        raise FileNotFoundError("Dataset not found")
    
    meas_records = np.array([utils.read_matrix_v2(x) for x in df['measurement_samples'].values])
    conditions = np.array([utils.read_matrix_v2(x) for x in df['coupling_matrix'].values])

    meas_records = meas_records.reshape(-1, shots_num, qubits_num)
    meas_records = meas_records.reshape(-1, qubits_num)

    new_conditions = []
    for i in range(samples_num):
        for _ in range(shots_num):
            new_conditions.append(conditions[i])
    new_conditions = np.array(new_conditions)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 512
    seq_len = qubits_num + 1 # +1 for the CLS token

    batch_conditions, batch_measures = [], []
    sample_idx = np.random.choice(range(samples_num*shots_num), batch_size, replace=False)
    batch_conditions = new_conditions[sample_idx]
    batch_measures = meas_records[sample_idx]
    all_embeddings, token_embedding = embedding.get_embedding(batch_size, seq_len, embedding_dim, batch_measures, batch_conditions)

    embeddings = all_embeddings
    labels = F.softmax(token_embedding, dim=-1)
    dataset = TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    

    decoder = Decoder(embedding_dim, seq_len, embedding_dim, ffn_hidden=128, n_head=8, n_layers=qubits_num, drop_prob=0.1, device='cuda')
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    
    epochs = 1000
    for epoch in tqdm(range(epochs)):
        decoder.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            trg_mask = torch.ones(8, all_embeddings.shape[1], all_embeddings.shape[1])
            trg_mask = torch.triu(trg_mask, diagonal=1)
            trg_mask = trg_mask.masked_fill(trg_mask == 1, float(0))

            outputs = decoder(inputs, trg_mask)

            loss = criterion(outputs.contiguous().view(-1, all_embeddings.size(-1)), targets.contiguous().view(-1, all_embeddings.size(-1)))
 
            gradients = torch.autograd.grad(loss, decoder.parameters(), retain_graph=True)
            for param, grad in zip(decoder.parameters(), gradients):
                param.grad = grad
            # loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.8f}')

    # save the decoder model
    pretrain_path = "save/pretrain_q{}_s{}_bs{}_ep{}.pt".format(qubits_num, shots_num, batch_size, epochs)
    torch.save(decoder.state_dict(), pretrain_path)
    print("Pretrained model saved at", pretrain_path)
    
if __name__ == "__main__":
    qubits_list = [8, 10, 12]
    for qubits_num in qubits_list:
        main(qubits_num)