import yaml
import argparse
import time
from tqdm import tqdm

from transformers import get_polynomial_decay_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
import torch.optim as optim

import random
import torch
import math
import numpy as np

from net.model import GraphTransformerNet
from data.preprocess import GenerateGraphData
from data.utils import add_laplacian_pos_encoding, collate
from data.dataset import DrugDataset
from trainer_utils import evaluate_network, train_epoch


def set_seed(seed, device_type):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)


def main(net_params, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_enc_dim = net_params["pos_enc_dim"]

    train_data = GenerateGraphData(data_path=params['path2data'], dataset_name=params['dataname'],
                                   data_flag=params["trainflag"])
    val_data = GenerateGraphData(data_path=params['path2data'], dataset_name=params['dataname'],
                                 data_flag=params["valflag"])
    test_data = GenerateGraphData(data_path=params['path2data'], dataset_name=params['dataname'],
                                  data_flag=params["testflag"])
    train_graphs, val_graphs, test_graphs = add_laplacian_pos_encoding(train_data.graph_list, val_data.graph_list,
                                                                       test_data.graph_list, pos_enc_dim)
    train_labels, val_labels, test_labels = train_data.pk_list, val_data.pk_list, test_data.pk_list

    trainset = DrugDataset(train_graphs, train_labels)
    valset = DrugDataset(val_graphs, val_labels)
    testset = DrugDataset(test_graphs, test_labels)

    set_seed(params['seed'], device.type)

    print("Training Graphs: ", len(train_labels))
    print("Validation Graphs: ", len(val_labels))
    print("Test Graphs: ", len(test_labels))

    model = GraphTransformerNet(net_params)
    model.to(device)

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate)

    if params['use_warmup']:
        optimizer = AdamW(model.parameters(), lr=params['init_lr'])
        num_update_steps_per_epoch = len(train_loader)
        max_steps = math.ceil(params['epochs'] * num_update_steps_per_epoch)
        num_warmup_steps = int(
            params['length_warmup'] * num_update_steps_per_epoch)  # Neccessary Number of steps to go from 0.0 to lr
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, max_steps,
                                                              lr_end=params['min_lr'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         verbose=True)

    best_val_rmse = float('inf')
    ckpt_dir = params["ckpt_dir"]

    with tqdm(range(params['epochs'])) as t:
        for epoch in t:

            t.set_description('Epoch %d' % epoch)

            start = time.time()

            epoch_train_loss, epoch_train_rmse, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                        scheduler=scheduler if params[
                                                                            'use_warmup'] else None)

            epoch_val_loss, epoch_val_rmse = evaluate_network(model, device, val_loader, epoch)
            _, epoch_test_rmse = evaluate_network(model, device, test_loader, epoch)

            t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                          train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                          train_RMSE=epoch_train_rmse, val_RMSE=epoch_val_rmse,
                          test_RMSE=epoch_test_rmse)

            if epoch_val_rmse < best_val_rmse:
                print("Hooray, New Best Validation")
                best_val_rmse = epoch_val_rmse
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + f"/{epoch}_{str(epoch_val_rmse)}_"
                                                                          f"{str(epoch_test_rmse)}"))

            if not params['use_warmup']:
                scheduler.step(epoch_val_rmse)
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET. FINISH TRAINING")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    args = parser.parse_args()
    path2config = args.config_path

    with open(path2config) as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)

    net_params, params = config['net_params'], config['params']

    main(net_params, params)




