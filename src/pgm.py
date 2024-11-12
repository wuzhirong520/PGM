# -*- coding: utf-8 -*-
#

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

    model = GINPredictor(num_node_emb_list=[119, 4],
                         num_edge_emb_list=[6, 3],
                         num_layers=2,
                         emb_dim=300,
                         JK='last',
                         dropout=0.2,
                         readout='mean',
                         n_tasks=dataset.n_tasks)
    model.to(device)
    model.zero_grad()

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

    count = 0
    for epoch in range(0, args.num_epochs):
        progress_bar.update(1)
        train_meter = Meter()
        for batch_id, batch_data in enumerate(train_loader):
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

            new_model = GINPredictor(num_node_emb_list=[119, 4],
                                     num_edge_emb_list=[6, 3],
                                     num_layers=2,
                                     emb_dim=300,
                                     JK='last',
                                     dropout=0.2,
                                     readout='mean',
                                     n_tasks=dataset.n_tasks)

            new_model.train()
            new_model.to(device)
            new_model.zero_grad()
            logits = new_model(bg, node_feats, edge_feats)
            loss = (criterion(logits, labels) * (masks != 0).float()).mean()

            logs = {"loss":loss.detach().cpu()}
            progress_bar.set_postfix(**logs)
            if args.wandb is not None:
                wandb.log(logs)

            loss.backward()

            for p, new_p in zip(model.parameters(), new_model.parameters()):
                p.data = (p.data * count + new_p.grad) / (count + 1)
            count += 1

            train_meter.update(logits, labels, masks)

        if (epoch + 1) % args.save_ckpt_step == 0:
            save_dir = os.path.join(str(args.log_path), args.run_name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({'epoch': epoch, 'state': model.state_dict()}, os.path.join(save_dir, str(args.dataset) + f'_PGM_{epoch + 1:05}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Principal gradient calculation')
    parser.add_argument('--split', type=str, default='scaffold', choices=['scaffold', 'random'],
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('--split-ratio', type=str, default='1,0,0',
                        help='Proportion of the dataset to use for training, validation and test (default: 1,0,0)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')

    parser.add_argument('--dataset', type=str, required=True, choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                'ToxCast', 'HIV', 'PCBA', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL'],
                    help='Dataset to use')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Maximum number of epochs for calculating PGM (default: 10)')
    parser.add_argument('--seed', type=int, default=24012128,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--save_ckpt_step", type=int , default=1)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument("--wandb", action='store_true')

    args = parser.parse_args()
    print(args)
    os.makedirs(args.log_path, exist_ok=True)

    if args.wandb is not None:
        wandb.init(project="PGM", name=args.run_name, config=args)

    main(args)

    if args.wandb is not None:
        wandb.finish()