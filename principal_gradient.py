# -*- coding: utf-8 -*-
#

import os
import argparse
import torch
import torch.nn as nn
from dgllife.utils import PretrainAtomFeaturizer
from dgllife.utils import PretrainBondFeaturizer
from dgllife.utils import Meter, SMILESToBigraph
from dgllife.model.model_zoo.gin_predictor import GINPredictor
from torch.utils.data import DataLoader
from utils import split_dataset, collate_molgraphs


def main(args, dataset, device):
    train_set, val_set, test_set = split_dataset(args, dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True,
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

    if args.task == 'classification':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif args.task == 'regression':
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError("Invalid task type.")

    count = 0
    for epoch in range(0, args.num_epochs):
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
            loss.backward()

            for p, new_p in zip(model.parameters(), new_model.parameters()):
                p.data = (p.data * count + new_p.grad) / (count + 1)
            count += 1

            train_meter.update(logits, labels, masks)

        torch.save({'epoch': epoch, 'state': model.state_dict()}, os.path.join(str(args.result_path), str(args.dataset) + '_' + str(epoch+1) + 'epoch_grads.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Principal gradient calculation')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any. (default: 0)')
    parser.add_argument('-d', '--dataset', type=str, default='BACE', choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL'],
                        help='Dataset to use')
    parser.add_argument('-s', '--split', type=str, default='scaffold', choices=['scaffold', 'random'],
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', type=str, default='1,0,0',
                        help='Proportion of the dataset to use for training, validation and test (default: 1,0,0)')
    parser.add_argument('-t', '--task', type=str, default='classification', choices=['classification', 'regression'],
                        help='task type (default: classification)')
    parser.add_argument('-n', '--num-epochs', type=int, default=10,
                        help='Maximum number of epochs for calculating PGM (default: 10)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-rp', '--result_path', type=str, default='results')
    args = parser.parse_args

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
