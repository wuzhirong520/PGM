# -*- coding: utf-8 -*-
#
# adapted from
# https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/moleculenet/classification.py

import os
os.environ["OMP_NUM_THREADS"] = "1" # noqa
os.environ["MKL_NUM_THREADS"] = "1" # noqa
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


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_dataset(args.dataset)
    print("Dataset size: ", len(dataset))

    train_set, val_set, test_set = split_dataset(args, dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,
                            collate_fn=collate_molgraphs, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size,
                             collate_fn=collate_molgraphs, num_workers=args.num_workers)

    model = GINPredictor(num_node_emb_list=[119, 4],
                         num_edge_emb_list=[6, 3],
                         num_layers=2,
                         emb_dim=300,
                         JK='last',
                         dropout=0.2,
                         readout='mean',
                         n_tasks=dataset.n_tasks)
    if args.load_ckpt_path is not None:
        pretrained = torch.load(args.load_ckpt_path)
        model_dict = model.state_dict()
        model_dict.update({k: v for k, v in pretrained.items() if 'gnn' in k})
        model.load_state_dict(model_dict)
    model.to(device)
    print(model)

    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters(), "lr": (0 if args.freeze_gnn else args.learning_rate)})
    model_param_group.append({"params": model.predict.parameters(), "lr": args.learning_rate})
    optimizer = torch.optim.Adam(model_param_group, lr=args.learning_rate, weight_decay=0)

    if DATASET_INFO[args.dataset]['task'] == 'classification':
        score_inv= 1.0
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif DATASET_INFO[args.dataset]['task'] == 'regression':
        score_inv= -1.0
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError("Invalid task type.")

    progress_bar = tqdm(
        range(0, args.num_epochs),
        desc="Steps"
    )

    best_valid = -999999
    for epoch in range(0, args.num_epochs):
        progress_bar.update(1)
        train_score = train(args, epoch, model, train_loader, criterion, optimizer, device) * score_inv
        val_score = evaluation(args, model, val_loader, device) * score_inv
        if val_score > best_valid:
            best_valid = val_score
        test_score = evaluation(args, model, test_loader, device) * score_inv

        logs = {"train_score":train_score, "val_score":val_score, "best_val":best_valid, "test_score":test_score}
        progress_bar.set_postfix(**logs)
        if args.wandb:
            wandb.log(logs)

        if (epoch + 1) % args.save_ckpt_step == 0:
            save_dir = os.path.join(str(args.log_path), args.run_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, str(args.dataset) + f'_model_{epoch + 1:05}.pt'))
            if val_score==best_valid:
                torch.save(model.state_dict(), os.path.join(save_dir, str(args.dataset) + f'_model.pt'))
                with open(os.path.join(save_dir, "best_val_score.txt"),"w") as f:
                    f.write(str(best_valid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training')
    parser.add_argument('--split', type=str, default='scaffold', choices=['scaffold', 'random'],
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('--split-ratio', type=str, default='0.8,0.1,0.1',
                        help='Proportion of the dataset to use for training, validation and test (default: 0.8,0.1,0.1)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')

    parser.add_argument('--dataset', type=str, required=True, choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL'], help='Dataset to use')
    parser.add_argument('--seed', type=int, default=24012128,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='Maximum number of epochs for pre-training (default: 300)')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--save_ckpt_step", type=int , default=1)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument("--load_ckpt_path", type=str)
    parser.add_argument("--freeze_gnn",type=bool, default=False)
    parser.add_argument("--wandb", action='store_true')

    args = parser.parse_args()
    print(args)
    os.makedirs(args.log_path, exist_ok=True)

    
    if args.wandb:
        wandb.init(project="PGM", name=args.run_name, config=args)

    main(args)

    if args.wandb:
        wandb.finish()
