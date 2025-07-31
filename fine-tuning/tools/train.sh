CONFIG=$1  # 第一个参数是 config 路径
PY_ARGS=${@:2}  # 其余参数作为额外传给训练脚本的参数（如 --work-dir 等）

# 指定使用 GPU0 并运行训练脚本
CUDA_VISIBLE_DEVICES=0\
PYTHONPATH="/homes/yunhengwu/code/D1/CLSM_FM/dense_prediction:$PYTHONPATH" \
python $(dirname "$0")/train.py \
    $CONFIG \
    --launcher none \
    $PY_ARGS \
    --resume