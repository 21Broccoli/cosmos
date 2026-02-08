**几何同变 Action–Video Pipeline（结合现有 Cosmos 代码）**

1. **数据解析与几何预处理**  
   - 新增脚本 `cosmos_policy/datasets/utils/create_action_flow.py`（或直接扩展 `datasets/aloha_dataset.py`）：  
     - 读取 `observation/*_camera/{rgb,intrinsic_cv,extrinsic_cv,cam2world_gl}`、`joint_action/vector`、`endpose/{left,right}_endpose`。  
     - 计算每帧末端 SE(3) 位姿 `T_base→ee(t)` 及增量 `ΔT(t)`；选取关键点/接触点（可用 `pointcloud`）。  
     - 投影到像素：`proj(u,v) = K * [R|t] * X`，得到每帧关键点的像素位置与位移 `Δ(u,v)`。  
     - 输出 `F_action`（与 VAE latent 目标分辨率一致的向量场，比如 32×32×2）和 `F_mask`（受动作影响区域），缓存到 `.npy` 或直接写进 HDF5（例如 `augmentation/action_flow` 数据集）。  
   - 保证 WAN tokenizer 仍只接收 RGB → latent（不改 WAN 本体）。

2. **latent 注入逻辑修改**  
   - 文件 `cosmos_policy/models/policy_text2world_model.py`:  
     - 在 `replace_latent_with_action_chunk` 中，把输入的动作从 `[B, chunk_size, action_dim]` 转为 `[B, C_flow, H', W']`（来自上一步 `F_action`）。  
     - 增加新函数 `replace_latent_with_action_flow`，将 `F_action`（和可选 `F_mask`）写入 `action_latent_idx` 对应帧；如需保留原 14 维动作，可新增 `action_scalar_latent_idx` slot。  
     - 在 `compute_loss_with_epsilon_and_sigma` 里将 `world_model_sample_mask`/`action_latent_idx` 对应的 target 改为 `F_action`，同时把原 `action_chunk` loss 保留在新的 scalar slot。  
   - 配置 `ConfigV2`：在相关 experiment config（例如 `config/experiment/cosmos_predict2_2b_480p_aloha_...`）里添加 flag（`use_action_flow=True`、`action_flow_channels=2`），用于控制是否启用此机制。

3. **Flow Matching / Loss 扩展**  
   - 在 `cosmos_policy/models/policy_text2world_model.py` 的 `compute_loss_with_epsilon_and_sigma` 中：  
     - 计算视频光流 `F_video`：可在 dataset 预处理时生成，也可以在训练时通过相邻帧 latent 差求得（`Δlatent = x_t+1 - x_t`）。  
     - 新增 `flow_matching_loss = flow_loss(F_video, action_flow_pred)`；你可以创建模块 `cosmos_policy/modules/flow_matching.py`，提供 `compute_flow_loss(flow_gt, flow_pred, mask, loss_type="sinkhorn")`。  
     - 将此 loss 乘以权重（例如配置项 `trainer.loss_weights.flow_matching`）加入最终 loss，并在日志中记录。

4. **MiniTrainDIT / Action DiT 调整**  
   - 文件 `cosmos_policy/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py`:  
     - 在 `ActionConditionedMinimalV1LVGDiT.forward` 里，新增 `action_flow` / `action_mask` 输入，根据 `data_batch["action_flow"]` 直接叠加到 `x_B_C_T_H_W` 对应通道。  
     - 若要实现 Flow Alignment Head，可在 `MiniTrainDIT` block（`cosmos_policy/_src/predict2/networks/minimal_v4_dit.py`）里给 `self.blocks` 增加可插拔模块：  
       1) 在 block 初始化时根据配置创建 `FlowAlignmentHead`；  
       2) 在 `forward` 中对 action token & video token 做一次 cross-attn，输出的偏置项加回 action token。  
     - 配置项在 `config/experiment/...` 中加入 `model.config.enable_flow_alignment=True` 等 flag。

5. **Dataset Loader 改动**  
   - 在 `datasets/aloha_dataset.py`:  
     - `__getitem__` 返回 dict 中新增 `action_flow`, `action_flow_mask`, `video_flow_gt`。  
     - 如果 Flow 预处理数据较大，可按需读取；可在 `__init__` 里通过路径（例如 `action_flow_dir`) 指定外部缓存。  
     - 确保 `collate_fn` 与 `DataLoader` 支持这些新字段。

6. **推理流程 `cosmos_utils.get_action`**  
   - 在构建 `image_sequence` 前，使用当前观测估计 `F_action`：  
     - 可用上一帧末端位姿 + 预测动作输出近似 `ΔT`，再投影形成占位向量场（或简单假设零流）。  
     - 在 `data_batch` 中添加 `action_flow_placeholder` / `flow_mask_placeholder`，模型生成后在 `action_latent_idx` 提取 `F_action_pred`，再通过一个 MLP/解码器（可新增 `ActionFlowDecoder` 模块）还原为 14 维动作。  
   - 解码器可放在 `cosmos_policy/experiments/robot/cosmos_utils.py`，例如 `decode_action_from_flow(flow_latent, decoder)`，保持 WAN 主体不变。

7. **兼容 WAN 预训练权重**  
   - 所有改动发生在：  
     1) Dataset 输出；  
     2) `policy_text2world_model.py` 注入/损失；  
     3) `ActionConditionedMinimal...` 中添加额外输入与 Head；  
     4) 新增 decoder/flow loss 模块。  
   - WAN VAE 本身不动（仍三通道 RGB），DiT 模型结构可通过 adapter 插入（例如 Flow Alignment Head 只加在 residual 上，不破坏原权重维度），因此加载 HuggingFace 预训练权重时，只需对新增参数设随机初始化，再继续 fine-tune 即可。

**实施顺序建议**

1. **Stage A（数据 & 注入）**  
   - 扩展 dataset + latent 注入，实现 `F_action` 写入 action slot，保持其他部分不变，确认模型能训练、推理。  
2. **Stage B（Flow Loss）**  
   - 加 `flow_matching_loss`，确保 training loop、日志都 OK。  
3. **Stage C（Flow Alignment Head / Decoder 优化）**  
   - 根据需要再改 `MiniTrainDIT` block，并实现动作向量场→原动作的 decoder，以便推理阶段恢复控制命令。

这样每一步都清楚地对应 Cosmos 代码中的文件/函数，不影响 WAN 预训练的主体结构，同时实现你希望的“动作 latent 与视频 latent 同构/同变 + Flow Matching”目标。需要我在某一步写具体示例代码或 patch，也可以继续告诉我。