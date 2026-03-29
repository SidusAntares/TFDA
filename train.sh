#!/usr/bin/env bash
set -euo pipefail  # 启用严格模式：出错即停
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/tfda/bin/python"
# ================== 配置区（集中管理，便于修改）==================
SOURCES=("FR1" "FR2" "AT1" "DK1")
# "france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017"
SEEDS=(111)
TYPE=('head')

# ================================================================


# 执行单个训练任务的函数
run_experiment() {
    local source_path="$1"


#    local seed="$2"
    echo "--------------------------------------------------"
    echo "[INFO] 开始训练: source=$source_path"
    echo "[CMD] "$PYTHON_EXEC" train.py --dataset '$source_path' "
    echo "--------------------------------------------------"

    # 执行命令，失败则退出
    "$PYTHON_EXEC" ./trainers/train.py --dataset "$source_path"
#    "$PYTHON_EXEC" ./process.py --source "$source_path" --target "$source_path"
}

# 主循环：遍历所有组合
for source in "${SOURCES[@]}"; do
    run_experiment "$source"
done

echo "[SUCCESS] 所有实验已完成！共 $((${#SOURCES[@]} * (${#SOURCES[@]} -1) * ${#SEEDS[@]})) 个任务。"

