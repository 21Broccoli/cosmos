# Cosmos 几何同变 Action–Video Pipeline 命令

## 0. 环境准备与路径约定
```bash
cd /home/jiaruiy/cosmos-policy
uv sync --extra cu128 --group aloha --python 3.10

export BASE_DATASETS_DIR=/mnt/nvme3/jiaruiy/cosmos_datasets
export ALOHA_RAW_ROOT=/mnt/nvme3/jiaruiy/aloha_agilex_clean_50
export ALOHA_PREPROC_DIR=$BASE_DATASETS_DIR/ALOHA-Cosmos-Policy
export COSMOS_ACTION_FLOW_DATASET_ROOT=$ALOHA_RAW_ROOT
export COSMOS_ACTION_FLOW_ROOT=/mnt/nvme3/jiaruiy/cosmos
export COSMOS_JOB_ROOT=/mnt/nvme3/jiaruiy/cosmos_policy_jobs
export ALOHA_MIXTURE_NAME=turn_switch        # 修改为你当前训练/推理所用的预处理任务混合
export ALOHA_MIXTURE_DIR=$ALOHA_PREPROC_DIR/$ALOHA_MIXTURE_NAME/data
export ALOHA_T5_EMB=$ALOHA_MIXTURE_DIR/t5_embeddings.pkl
export ALOHA_DATASET_STATS=$ALOHA_MIXTURE_DIR/dataset_statistics.json
export ALOHA_DATA_DIR=$ALOHA_MIXTURE_DIR     # 训练配置会优先读取 ALOHA_DATA_DIR / ALOHA_MIXTURE_DIR
mkdir -p "$BASE_DATASETS_DIR" "$ALOHA_PREPROC_DIR" "$COSMOS_ACTION_FLOW_ROOT" "$COSMOS_JOB_ROOT"
```
（可选）直接同步官方预处理数据供基线对比：
```bash
hf download nvidia/ALOHA-Cosmos-Policy --repo-type dataset --local-dir "$BASE_DATASETS_DIR/ALOHA-Cosmos-Policy"
```

## 1. 生成几何同变 Action Flow 缓存
首先，写缓存：
```bash

bash -lc '
src=/mnt/nvme3/jiaruiy/aloha_agilex_clean_50
dst=/mnt/nvme3/jiaruiy/cosmos
max_tasks=${MAX_TASKS:-30}   # export MAX_TASKS=-1 to process everything
mkdir -p "$dst"
cd /home/jiaruiy/cosmos-policy
count=0
for task_dir in "$src"/*; do
  [ -d "$task_dir" ] || continue
  count=$((count + 1))
  if [ "$max_tasks" -gt 0 ] && [ $count -gt "$max_tasks" ]; then
    break
  fi
  task_name=$(basename "$task_dir")
  for arm in left right; do
    out_dir="$dst/$task_name/arm_$arm"
    uv run -m cosmos_policy.datasets.utils.create_action_flow \
      --dataset-root "$task_dir" \
      --output-root "$out_dir" \
      --camera head_camera \
      --arm "$arm" \
      --resolution 32 \
      --sigma 1.5
  done
done
'


```


## 2. ALOHA 数据预处理与语言嵌入

将原始 480×640 ALOHA episode 统一切分/缩放到 Cosmos 256×256 预处理目录（默认遍历全部任务；如需子集可在循环里筛选，输出会写入 `$ALOHA_PREPROC_DIR/<task_name>`）：
```bash
for task_dir in "$ALOHA_RAW_ROOT"/*; do
  [ -d "$task_dir" ] || continue
  task_name=$(basename "$task_dir")
  dataset_dir="$task_dir/data"
  if ! ls "$dataset_dir"/episode*.hdf5 >/dev/null 2>&1; then
    echo "Skip $task_name: no episode*.hdf5 under $dataset_dir"
    continue
  fi
  out_dir="$BASE_DATASETS_DIR/ALOHA-Cosmos-Policy/$task_name"
  echo "Preprocessing $task_name"
  uv run --extra cu128 --group aloha --python 3.10 \
    python cosmos_policy/experiments/robot/aloha/preprocess_split_aloha_data.py \
    --dataset_path "$dataset_dir" \
    --out_base_dir "$out_dir" \
    --percent_val 0.01
done
```

预计算 T5 文本嵌入并生成最新统计文件（每个 `$ALOHA_PREPROC_DIR/<task>/data` 目录都会写出独立的 `t5_embeddings.pkl` 与 `dataset_statistics*.json`；请将目标 mixture 的文件路径赋值给 `ALOHA_T5_EMB` / `ALOHA_DATASET_STATS`）：
```bash
for task_dir in "$ALOHA_PREPROC_DIR"/*; do
  [ -d "$task_dir" ] || continue
  dataset_root="$task_dir/data"
  echo "Computing T5 embeddings for $(basename "$task_dir")"
  if [ ! -d "$dataset_root" ]; then
    echo "Skip $(basename "$task_dir"): missing data/ split"
    continue
  fi
  uv run --extra cu128 --group aloha --python 3.10 \
    -m cosmos_policy.datasets.save_aloha_t5_text_embeddings \
    --data_dir "$dataset_root"
done

python - <<'PY'
import os
from cosmos_policy.datasets.aloha_dataset import ALOHADataset

mixture_dir = os.environ["ALOHA_MIXTURE_DIR"]
ALOHADataset(
    data_dir=mixture_dir,
    t5_text_embeddings_path=os.environ["ALOHA_T5_EMB"],
    action_flow_root=os.environ["COSMOS_ACTION_FLOW_ROOT"],
    action_flow_dataset_root=os.environ["COSMOS_ACTION_FLOW_DATASET_ROOT"],
    action_flow_camera="head_camera",
    action_flow_include_video_gt=True,
)
PY
```

## 3. 训练（Action–Video 几何同变配置）
```bash
export FLOW_JOB_NAME=aloha_action_flow_cosmos2b
# 若只使用部分 GPU，可先设置 `export CUDA_VISIBLE_DEVICES=2,3` 并把 `--nproc_per_node` 改成当前可见显卡的数量。
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=2 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80" \
  +job.path_local="$COSMOS_JOB_ROOT/${FLOW_JOB_NAME}" \
  trainer.grad_accum_iter=16 \
  dataloader_train.batch_size=16 \
  model.config.flow_matching_loss_weight=0.15 \
  model.config.flow_action_loss_weight=0.1 \
  model.config.decode_actions_from_flow=True \
  model.config.enable_flow_alignment_head=True

```
> 说明：`COSMOS_ACTION_FLOW_ROOT` / `_DATASET_ROOT` 会被数据集自动读取，确保 flow/mask/video_flow_gt 字段被串联进训练管线。

## 4. 推理服务（部署 Cosmos Policy + Flow 解码）
```bash
export FLOW_POLICY_CKPT="$COSMOS_JOB_ROOT/${FLOW_JOB_NAME}/checkpoints/global_step50000/mp_rank_00_model_states.pt"  # 根据实际 checkpoint 名称修改
uv run --extra cu128 --group aloha --python 3.10 \
  -m cosmos_policy.experiments.robot.aloha.deploy \
  --config cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only \
  --ckpt_path "$FLOW_POLICY_CKPT" \
  --config_file cosmos_policy/config/config.py \
  --use_third_person_image True \
  --use_wrist_image True \
  --num_wrist_images 2 \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --dataset_stats_path "$ALOHA_DATASET_STATS" \
  --t5_text_embeddings_path "$ALOHA_T5_EMB" \
  --trained_with_image_aug True \
  --chunk_size 50 \
  --num_open_loop_steps 50 \
  --num_denoising_steps_action 10 \
  --num_denoising_steps_future_state 3 \
  --num_denoising_steps_value 1 \
  --deterministic True \
  --seed 7
```

## 5. 机器人端推理 / 闭环评测
```bash
uv run --extra cu128 --group aloha --python 3.10 \
  -m cosmos_policy.experiments.robot.aloha.run_aloha_eval \
  --policy_name cosmos_flow_pipeline_debug \
  --policy_server_ip 127.0.0.1 \
  --input_image_size 224 \
  --trained_with_image_aug True \
  --num_open_loop_steps 50 \
  --num_rollouts_planned 5 \
  --max_steps 1100 \
  --future_img True \
  --return_all_query_results True \
  --data_collection True \
  --run_id_note flow_alignment_debug \
  --local_log_dir experiments/robot/aloha/logs
```
> 在机器人端脚本运行时，会通过 `cosmos_utils.get_action()` 生成 `action_flow_pred`，经 `ActionFlowDecoder` 解码回 14 维控制量，实现与训练一致的几何同变闭环。
