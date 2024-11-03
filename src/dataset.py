import os
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, SMILESToBigraph

DATASET_INFO = {
    "BACE": {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "HIV": {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "MUV": {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "PCBA": {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "BBBP": {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "ClinTox": {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "SIDER": {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "Tox21": {
        "task":"classification",
        "metric":"roc_auc_score",
    } ,
    "ToxCast" : {
        "task":"classification",
        "metric":"roc_auc_score",
    },
    "ESOL": {
        "task":"regression",
        "metric":"rmse",
    },
    "FreeSolv": {
        "task":"regression",
        "metric":"rmse",
    },
    "Lipophilicity": {
        "task":"regression",
        "metric":"rmse",
    }
}

def load_dataset(dataset_name : str, bin_path = "./bin", load_from_bin = True,  num_workers = 1):
    atom_featurizer = PretrainAtomFeaturizer()
    bond_featurizer = PretrainBondFeaturizer()
    smiles_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer)
    bin_file = os.path.join(bin_path, dataset_name.lower() + "_dglgraph.bin")
    load = True if load_from_bin and os.path.exists(bin_file) else False
    if dataset_name == 'MUV':
        from dgllife.data import MUV
        dataset = MUV(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'BACE':
        from dgllife.data import BACE
        dataset = BACE(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'BBBP':
        from dgllife.data import BBBP
        dataset = BBBP(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'ClinTox':
        from dgllife.data import ClinTox
        dataset = ClinTox(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'SIDER':
        from dgllife.data import SIDER
        dataset = SIDER(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'ToxCast':
        from dgllife.data import ToxCast
        dataset = ToxCast(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'HIV':
        from dgllife.data import HIV
        dataset = HIV(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'PCBA':
        from dgllife.data import PCBA
        dataset = PCBA(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'Tox21':
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'FreeSolv':
        from dgllife.data import FreeSolv
        dataset = FreeSolv(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'Lipophilicity':
        from dgllife.data import Lipophilicity
        dataset = Lipophilicity(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    elif dataset_name == 'ESOL':
        from dgllife.data import ESOL
        dataset = ESOL(smiles_to_graph=smiles_to_g, cache_file_path=bin_file,load=load, n_jobs=num_workers)

    else:
        raise ValueError('Unexpected dataset: {}'.format(dataset_name))

    return dataset