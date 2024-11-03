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

PRETRAINED_CKPTS={
    "BACE":"/home/wuzhirong/AI4S/PGM/log/pretrain-BACE-1730625427/BACE_model_00200.pt",
    "HIV":"/home/wuzhirong/AI4S/PGM/log/pretrain-HIV-1730625484/HIV_model_00200.pt",
    "MUV":"/home/wuzhirong/AI4S/PGM/log/pretrain-MUV-1730626417/MUV_model_00200.pt",
    "PCBA":"/home/wuzhirong/AI4S/PGM/log/pretrain-PCBA-1730628569/PCBA_model_00078.pt",
    "BBBP":"/home/wuzhirong/AI4S/PGM/log/pretrain-BBBP-1730633844/BBBP_model_00200.pt",
    "ClinTox":"/home/wuzhirong/AI4S/PGM/log/pretrain-ClinTox-1730633910/ClinTox_model_00200.pt",
    "SIDER":"/home/wuzhirong/AI4S/PGM/log/pretrain-SIDER-1730633965/SIDER_model_00200.pt",
    "Tox21" :"/home/wuzhirong/AI4S/PGM/log/pretrain-Tox21-1730634075/Tox21_model_00200.pt",
    "ToxCast" :"/home/wuzhirong/AI4S/PGM/log/pretrain-ToxCast-1730634274/ToxCast_model_00200.pt",
    "ESOL":"/home/wuzhirong/AI4S/PGM/log/pretrain-ESOL-1730636043/ESOL_model_00200.pt",
    "FreeSolv":"/home/wuzhirong/AI4S/PGM/log/pretrain-FreeSolv-1730636083/FreeSolv_model_00200.pt",
    "Lipophilicity":"/home/wuzhirong/AI4S/PGM/log/pretrain-Lipophilicity-1730636116/Lipophilicity_model_00200.pt",
}


if __name__=="__main__":
    # for dataset in DATASET_NAMES:
    #     # if dataset=="MUV" or dataset=="BACE" or dataset=="PCBA" or dataset=="HIV"or dataset=="BBBP"or dataset=="ClinTox"or dataset=="SIDER"or dataset=="Tox21" or dataset=="ToxCast":
    #     #     continue
    #     cmd = "CUDA_VISIBLE_DEVICES=2 "
    #     cmd += "python src/pretrain.py "
    #     cmd += " --dataset " + dataset
    #     cmd += " --batch_size " + "32"
    #     # for k,v in TASK_CONFIGS[dataset].items():
    #     #     cmd += " --" + k + " " + str(v)
    #     os.system(cmd)
    for dataset in DATASET_NAMES:
        for base_dataset in DATASET_NAMES:
            if base_dataset==dataset:
                continue
            cmd = "CUDA_VISIBLE_DEVICES=3 "
            cmd += "python src/pretrain.py "
            cmd += " --dataset " + dataset
            cmd += " --mode finetune"
            cmd += " --finetune_base_dataset " + base_dataset
            cmd += " --load_ckpt_path " + PRETRAINED_CKPTS[base_dataset]
            os.system(cmd)