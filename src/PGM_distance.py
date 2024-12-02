import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np


def distance(grad1, grad2):
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
    return PGM_distance.item()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='PGM distance calculation')
    # parser.add_argument('-g1', '--grad1', type=str, help='filename of the saved principal gradient')
    # parser.add_argument('-g2', '--grad2', type=str, help='filename of the saved principal gradient')
    # args = parser.parse_args()
    grads = {
        "BACE"   :"/home/wuzhirong/AI4S/PGM/log/pgm-BACE-001/BACE_PGM_00010.pt",
        "BBBP"   :"/home/wuzhirong/AI4S/PGM/log/pgm-BBBP-001/BBBP_PGM_00010.pt",
        "ClinTox":"/home/wuzhirong/AI4S/PGM/log/pgm-ClinTox-001/ClinTox_PGM_00010.pt",
        "ESOL" : "/home/wuzhirong/AI4S/PGM/log/pgm-ESOL-001/ESOL_PGM_00010.pt",
        "FreeSolv":"/home/wuzhirong/AI4S/PGM/log/pgm-FreeSolv-001/FreeSolv_PGM_00010.pt",
        "HIV":"/home/wuzhirong/AI4S/PGM/log/pgm-HIV-001/HIV_PGM_00010.pt",
        "Lipophilicity":"/home/wuzhirong/AI4S/PGM/log/pgm-Lipophilicity-001/Lipophilicity_PGM_00010.pt",
        "MUV":"/home/wuzhirong/AI4S/PGM/log/pgm-MUV-001/MUV_PGM_00010.pt",
        "PCBA":"/home/wuzhirong/AI4S/PGM/log/pgm-PCBA-001/PCBA_PGM_00010.pt",
        "SIDER":"/home/wuzhirong/AI4S/PGM/log/pgm-SIDER-001/SIDER_PGM_00010.pt",
        "Tox21":"/home/wuzhirong/AI4S/PGM/log/pgm-Tox21-001/Tox21_PGM_00010.pt",
        "ToxCast":"/home/wuzhirong/AI4S/PGM/log/pgm-ToxCast-001/ToxCast_PGM_00010.pt"
    }
    # datasets = list(grads.keys())
    datasets = ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER','ToxCast', 'HIV', 'PCBA', 'Tox21', 'FreeSolv', 'Lipophilicity', 'ESOL']

    scale = 10000.0

    n = len(grads)

    tranfer_map = np.zeros((n,n))
    for i ,(dataset1) in enumerate(datasets):
        for j, (dataset2) in enumerate(datasets):
            if i==j:
                continue
            tranfer_map[i,j] = distance(grads[dataset1], grads[dataset2])*scale

    print([list(tranfer_map[i]) for i in range(12)])

    ranks = []
    for i in range(n):
        sorted_indices = tranfer_map[i].argsort() 
        ranks.append(list(sorted_indices))
    
    [print([datasets[j] for j in r]) for r in ranks]
    print("")

    print(ranks)

    plt.figure(figsize=(9, 9))
    plt.imshow(tranfer_map, cmap='viridis', interpolation='nearest')
    plt.xticks(range(n), [i[:5]+"\n"+i[5:] for i in datasets]) 
    plt.yticks(range(n), [i for i in datasets])
    plt.colorbar()
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{tranfer_map[i,j]:.2f}", ha='center', va='center', color='white')
    plt.title(f"Heatmap of the transfer map (X{1/scale})")
    plt.show()