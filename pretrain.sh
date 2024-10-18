python src/pretrain.py \
    --device 0 \
    --dataset "BACE" \
    --seed 0 \
    --task "classification" \
    --metric "roc_auc_score" \
    --num-epochs 300 \
    --num-workers 0 \
    --result_path "./log"