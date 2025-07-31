export PYTHONPATH=/homes/yunhengwu/code/D1/CLSM_FM/dense_prediction:$PYTHONPATH
CONFIG=$1           # 第一个参数：config 路径
CHECKPOINT=$2       # 第二个参数：checkpoint 路径
PY_ARGS=${@:3}      # 第三个参数及之后：其他参数

# 指定使用 GPU0 并运行测试脚本
CUDA_VISIBLE_DEVICES=7\
PYTHONPATH="/homes/yunhengwu/code/D1/CLSM_FM/dense_prediction:$PYTHONPATH" \
python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher none \
    $PY_ARGS


