import torch
import argparse


def main(grad1, grad2):
    dict1 = torch.load(grad1)
    dict2 = torch.load(grad2)
    grads1 = dict1['state']
    grads2 = dict2['state']

    count = 0
    for (k1, v1), (k2, v2) in zip(grads1.items(), grads2.items()):
        if 'gnn' in k1 and 'gnn' in k2:
            assert k1 == k2
            w1 = v1.flatten()
            w2 = v2.flatten()
            if count == 0:
                g1 = w1
                g2 = w2
            else:
                g1 = torch.cat([g1, w1], 0)
                g2 = torch.cat([g2, w2], 0)
            count += 1

    PGM_distance = torch.norm(g1 - g2) / (torch.norm(g1) * torch.norm(g2))
    print('PGM_distance: {}'.format(PGM_distance.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGM distance calculation')
    parser.add_argument('-g1', '--grad1', type=str, help='filename of the saved principal gradient')
    parser.add_argument('-g2', '--grad2', type=str, help='filename of the saved principal gradient')
    args = parser.parse_args()

    main(args.grad1, args.grad2)