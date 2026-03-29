#!/usr/bin/env bash
set -euo pipefail  # 启用严格模式：出错即停
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONDA_BASE=$(conda info --base)
PYTHON_EXEC="$CONDA_BASE/envs/tfda/bin/python"
# ================== 配置区（集中管理，便于修改）==================
SOURCES=("france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017")
# "france/31TCJ/2017" "france/30TXT/2017" "denmark/32VNH/2017" "austria/33UVP/2017"
SEEDS=(111)
TYPE=('head')

# ================================================================

# 执行单个训练任务的函数
run_experiment() {
    local source_path="$1"
    local target="$2"
    local seed="$3"

#    local seed="$2"
    echo "--------------------------------------------------"
    echo "[INFO] 开始训练: source=$source_path, target=$target, seed=$seed"
    echo "[CMD] "$PYTHON_EXEC" train.py --source '$source_path' --target '$target'"
    echo "--------------------------------------------------"

    # 执行命令，失败则退出
#    "$PYTHON_EXEC" ./train.py --source "$source_path" --target "$target" --seed "$seed"
    "$PYTHON_EXEC" ./process.py --source "$source_path" --target "$target" --seed "$seed"
}

# 主循环：遍历所有组合
for source in "${SOURCES[@]}"; do
    for target in "${SOURCES[@]}"; do
        if [[ "$source" == "$target" ]]; then
            continue
        fi
#        if [[ "$source" == "france/31TCJ/2017" && "$target" == "france/30TXT/2017" ]]; then
#            continue
#        fi
        for seed in "${SEEDS[@]}"; do
            run_experiment "$source"  "$target" "$seed"
        done
    done
done
#for source in "${SOURCES[@]}"; do
#
#        for seed in "${SEEDS[@]}"; do
#            run_experiment "$source"   "$seed"
#        done
#
#done

echo "[SUCCESS] 所有实验已完成！共 $((${#SOURCES[@]} * (${#SOURCES[@]} -1) * ${#SEEDS[@]})) 个任务。"

