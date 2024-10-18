# -*- coding: utf-8 -*-
#
# adapted from
# https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/moleculenet/classification.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from dgllife.utils import PretrainAtomFeaturizer
from dgllife.utils import PretrainBondFeaturizer
from dgllife.utils import Meter, SMILESToBigraph
from dgllife.model.model_zoo.gin_predictor import GINPredictor
from torch.utils.data import DataLoader
from utils import split_dataset, collate_molgraphs


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

    train_score = np.mean(train_meter.compute_metric(args.metric))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(epoch + 1, args.num_epochs, args.metric, train_score))


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
    return np.mean(eval_meter.compute_metric(args.metric))


def main(args, dataset, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set, val_set, test_set = split_dataset(args, dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=32,
                            collate_fn=collate_molgraphs, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=32,
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
    if args.task == 'classification':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif args.task == 'regression':
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError("Invalid task type.")

    best_valid = 0
    for epoch in range(0, args.num_epochs):
        train(args, epoch, model, train_loader, criterion, optimizer, device)
        val_score = evaluation(args, model, val_loader, device)
        if val_score > best_valid:
            best_valid = val_score
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(epoch + 1, args.num_epochs, args.metric, val_score, args.metric, best_valid))
        test_score = evaluation(args, model, test_loader, device)
        print('epoch {:d}/{:d}, test {} {:.4f}'.format(epoch + 1, args.num_epochs, args.metric, test_score))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(str(args.result_path), str(args.dataset) + f'_model_{epoch + 1}epoch.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any. (default: 0)')
    parser.add_argument('-d', '--dataset', type=str, default='BACE', choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('-s', '--split', type=str, default='scaffold', choices=['scaffold', 'random'],
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', type=str, default='0.8,0.1,0.1',
                        help='Proportion of the dataset to use for training, validation and test (default: 0.8,0.1,0.1)')
    parser.add_argument('-t', '--task', type=str, default='classification', choices=['classification', 'regression'],
                        help='task type (default: classification)')
    parser.add_argument('-me', '--metric', type=str, default='roc_auc_score', choices=['roc_auc_score', 'rmse'],
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=300,
                        help='Maximum number of epochs for pre-training (default: 300)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-rp', '--result_path', type=str, default='results')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    atom_featurizer = PretrainAtomFeaturizer()
    bond_featurizer = PretrainBondFeaturizer()
    smiles_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer)

    if args.dataset == 'MUV':
        from dgllife.data import MUV
        dataset = MUV(smiles_to_graph=smiles_to_g,
                      n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'BACE':
        from dgllife.data import BACE
        dataset = BACE(smiles_to_graph=smiles_to_g,
                       n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'BBBP':
        from dgllife.data import BBBP
        dataset = BBBP(smiles_to_graph=smiles_to_g,
                       n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'ClinTox':
        from dgllife.data import ClinTox
        dataset = ClinTox(smiles_to_graph=smiles_to_g,
                          n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'SIDER':
        from dgllife.data import SIDER
        dataset = SIDER(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'ToxCast':
        from dgllife.data import ToxCast
        dataset = ToxCast(smiles_to_graph=smiles_to_g,
                          n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'HIV':
        from dgllife.data import HIV
        dataset = HIV(smiles_to_graph=smiles_to_g,
                      n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'PCBA':
        from dgllife.data import PCBA
        dataset = PCBA(smiles_to_graph=smiles_to_g,
                       n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'Tox21':
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'FreeSolv':
        from dgllife.data import FreeSolv
        dataset = FreeSolv(smiles_to_graph=smiles_to_g,
                           n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'Lipophilicity':
        from dgllife.data import Lipophilicity
        dataset = Lipophilicity(smiles_to_graph=smiles_to_g,
                                n_jobs=1 if args.num_workers == 0 else args.num_workers)

    elif args.dataset == 'ESOL':
        from dgllife.data import ESOL
        dataset = ESOL(smiles_to_graph=smiles_to_g,
                       n_jobs=1 if args.num_workers == 0 else args.num_workers)

    else:
        raise ValueError('Unexpected dataset: {}'.format(args.dataset))


    main(args, dataset, device)
