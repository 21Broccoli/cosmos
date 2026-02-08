首先，写缓存：
```bash

bash -lc '
src=/mnt/nvme3/jiaruiy/aloha_agilex_clean_50
dst=/mnt/nvme3/jiaruiy/cosmos
mkdir -p "$dst"
cd /home/jiaruiy/cosmos-policy
count=0
for task_dir in "$src"/*; do
  [ -d "$task_dir" ] || continue
  count=$((count + 1))
  [ $count -gt 30 ] && break
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