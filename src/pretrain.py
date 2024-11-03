# -*- coding: utf-8 -*-
#
# adapted from
# https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/moleculenet/classification.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from dgllife.utils import Meter
from dgllife.model.model_zoo.gin_predictor import GINPredictor
from torch.utils.data import DataLoader
from utils import split_dataset, collate_molgraphs
from tqdm import tqdm
from dataset import load_dataset, DATASET_INFO
import wandb


def train(args, epoch, model, data_loader, loss_criterion, optimizer, device):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data

        if len(smiles) == 1:
            continue

        labels, masks = labels.to(device), masks.to(device)
        bg = bg.to(device)
        node_feats = [
            bg.ndata.pop('atomic_number').to(device),
            bg.ndata.pop('chirality_type').to(device)
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(device),
            bg.edata.pop('bond_direction_type').to(device)
        ]
        logits = model(bg, node_feats, edge_feats)
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.update(logits, labels, masks)

    train_score = np.mean(train_meter.compute_metric(DATASET_INFO[args.dataset]['metric']))
    return train_score


def evaluation(args, model, data_loader, device):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            _, bg, labels, masks = batch_data
            labels = labels.to(device)
            bg = bg.to(device)
            node_feats = [
                bg.ndata.pop('atomic_number').to(device),
                bg.ndata.pop('chirality_type').to(device)
            ]
            edge_feats = [
                bg.edata.pop('bond_type').to(device),
                bg.edata.pop('bond_direction_type').to(device)
            ]
            logits = model(bg, node_feats, edge_feats)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(DATASET_INFO[args.dataset]['metric']))


def main(args, dataset, run_name, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set, val_set, test_set = split_dataset(args, dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=1024*4, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=1024*4,
                            collate_fn=collate_molgraphs, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=1024*4,
                             collate_fn=collate_molgraphs, num_workers=args.num_workers)

    model = GINPredictor(num_node_emb_list=[119, 4],
                         num_edge_emb_list=[6, 3],
                         num_layers=2,
                         emb_dim=300,
                         JK='last',
                         dropout=0.2,
                         readout='mean',
                         n_tasks=dataset.n_tasks)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    if DATASET_INFO[args.dataset]['task'] == 'classification':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif DATASET_INFO[args.dataset]['task'] == 'regression':
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError("Invalid task type.")

    progress_bar = tqdm(
        range(0, args.num_epochs),
        desc="Steps"
    )

    best_valid = 0
    for epoch in range(0, args.num_epochs):
        progress_bar.update(1)
        train_score = train(args, epoch, model, train_loader, criterion, optimizer, device)
        val_score = evaluation(args, model, val_loader, device)
        if val_score > best_valid:
            best_valid = val_score
        test_score = evaluation(args, model, test_loader, device)

        logs = {"train_score":train_score, "val_score":val_score, "best_val":best_valid, "test_score":test_score}
        progress_bar.set_postfix(**logs)
        wandb.log(logs)

        if (epoch + 1) % args.save_ckpt_step == 0:
            save_dir = os.path.join(str(args.log_path), run_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, str(args.dataset) + f'_model_{epoch + 1:05}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any. (default: 0)')
    parser.add_argument('-d', '--dataset', type=str, default='BACE', choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=24012128,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('-s', '--split', type=str, default='scaffold', choices=['scaffold', 'random'],
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', type=str, default='0.8,0.1,0.1',
                        help='Proportion of the dataset to use for training, validation and test (default: 0.8,0.1,0.1)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')

    parser.add_argument('-n', '--num-epochs', type=int, default=300,
                        help='Maximum number of epochs for pre-training (default: 300)')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument("-bs","--batch_size", type=int, default=4096)
    parser.add_argument("--save_ckpt_step", type=int , default=10)
    args = parser.parse_args()

    import time
    run_name = "pretrain-"+args.dataset+"-"+str((int)(time.time()))

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    wandb.init(project="PGM", name=run_name, config=args)

    dataset = load_dataset(args.dataset)

    main(args, dataset, run_name, device)
