import numpy as np
import torch.nn.functional as F
import torch


def MSE(scores, targets):
    MSE = F.mse_loss(scores, targets)
    return MSE.detach().item()


def train_epoch(model, optimizer, device, data_loader, scheduler=None):
    model.train()
    epoch_loss = 0
    epoch_train_mse = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
        sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
        sign_flip[sign_flip >= 0.5] = 1.0;
        sign_flip[sign_flip < 0.5] = -1.0
        batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)

        batch_scores = model(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mse += MSE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
        if scheduler is not None:
            scheduler.step()
    epoch_loss /= (iter + 1)
    epoch_train_mse /= (iter + 1)

    return epoch_loss, np.sqrt(epoch_train_mse), optimizer


def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mse = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mse += MSE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mse /= (iter + 1)

    return epoch_test_loss, np.sqrt(epoch_test_mse)