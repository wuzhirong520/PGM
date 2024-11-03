import os

DATASET_NAMES = ["BACE",
                "HIV",
                "MUV",
                "PCBA",
                "BBBP",
                "ClinTox",
                "SIDER",
                "Tox21" ,
                "ToxCast" ,
                "ESOL",
                "FreeSolv",
                "Lipophilicity"]

TASK_CONFIGS = {
    "BACE": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "HIV": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "MUV": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "PCBA": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "BBBP": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "ClinTox": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "SIDER": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "Tox21": {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    } ,
    "ToxCast" : {
        "device": 0,
        "seed": 0,
        "task":"classification",
        "metric":"roc_auc_score",
        "num-workers": 4,
        "num-epochs":300
    },
    "ESOL": {
        "device": 0,
        "seed": 0,
        "task":"regression",
        "metric":"rmse",
        "num-workers": 4,
        "num-epochs":300
    },
    "FreeSolv": {
        "device": 0,
        "seed": 0,
        "task":"regression",
        "metric":"rmse",
        "num-workers": 4,
        "num-epochs":300
    },
    "Lipophilicity": {
        "device": 0,
        "seed": 0,
        "task":"regression",
        "metric":"rmse",
        "num-workers": 4,
        "num-epochs":300
    }
}

if __name__=="__main__":
    for dataset in DATASET_NAMES:
        if dataset!="ClinTox":
            continue
        # if dataset=="MUV" or dataset=="BACE" or dataset=="PCBA":
        #     continue
        cmd = "CUDA_VISIBLE_DEVICES=3 "
        cmd += "python src/pretrain.py "
        cmd += " --dataset " + dataset
        for k,v in TASK_CONFIGS[dataset].items():
            cmd += " --" + k + " " + str(v)
        os.system(cmd)