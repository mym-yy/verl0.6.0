# Tree Search + GRPO Debug 全记录 (2026-04-06)

## 1. 项目全局概览

### 1.1 目标

在 verl 0.6.0 + vLLM V0 引擎中实现 **entropy-based tree search decoding**，用于 GRPO 训练。核心思路：

- 每个 prompt 生成时，如果当前 token 的 entropy 超过阈值，就在该位置分叉（branching），产生多条候选路径
- 最终收集所有叶子节点（leaf nodes）作为 GRPO 的候选 response
- 替代原来的 `n>1` 并行采样方案

### 1.2 方案选择：Plan A (`n=1` + tree search)

```
n=1: 每个 prompt 只发起 1 个初始序列
tree search: 在解码过程中，遇到高 entropy token 时分叉出 branching_factor=3 个子序列
max_tree_depth=3: 最多分叉 3 层
理论最大叶子数: 3^3 = 27 个叶子/prompt
```

### 1.3 关键数字解释

以实际运行配置为例：

| 参数 | 值 | 含义 |
|------|-----|------|
| `train_batch_size` | 96 | 每个训练 step 的 prompt 总数 |
| `n_gpus_per_node` | 4 | GPU 数量 |
| 每 GPU prompt 数 | 96 / 4 = **24** | 每个 worker 分到 24 个 prompt |
| `branching_factor` | 3 | 每次分叉产生 3 个子序列 |
| `max_tree_depth` | 3 | 最多分叉 3 层 |
| `entropy_threshold` | 0 | entropy > 0 就分叉（几乎每个 token 都分叉） |
| 每 prompt 最大节点数 | 1 + 3 + 9 + 27 = **40** | depth 0: 1, depth 1: 3, depth 2: 9, depth 3: 27 |
| 每 prompt 最大叶子数 | 3^3 = **27** | 只有最底层的叶子节点是候选 response |
| 每 GPU 最大叶子数 | 24 × 27 = **648** | 日志中看到的 642~648（因为有些路径提前结束） |
| 全局总叶子数 | 4 × ~646 = **~2584** | 4 个 GPU 的叶子数加起来 |

**日志数字对应关系**：
```
"outputs_len": 24          ← 每个 GPU 处理 24 个 prompt（96/4）
"out0_outputs_len": 40     ← 第一个 prompt 产生 40 个节点（1+3+9+27，满树）
"leaf_count": 27           ← 其中 27 个是叶子节点
Expanded batch: 24 prompts -> 646 leaf responses  ← 24个prompt展开为646个叶子response
tree/total_nodes: 957      ← 该GPU上所有prompt的节点总数
tree/leaf_nodes: 646       ← 该GPU上所有prompt的叶子总数
tree/avg_max_depth: 3.0    ← 平均最大深度 = 3（因为threshold=0，几乎全满树）
```

### 1.4 修改的文件总览

| 文件 | 修改内容 |
|------|---------|
| `vllm/engine/llm_engine.py` | 修复 tree decoding 参数读取（从 parent group 读取原始参数）；修复 finished 序列检查 |
| `vllm/sequence.py` | 修复无限递归：子请求 params 必须禁用 tree search |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 验证阶段禁用 tree search；叶子节点收集；batch 维度扩展；non_tensor_batch 扩展 |
| `verl/trainer/ppo/ray_trainer.py` | 训练 batch 维度对齐（global prompt indices 重建）；tree metrics 聚合 |
| `tree/run_qwen3-8b.sh` | `n=8` → `n=1`；`ppo_mini_batch_size=2` → `16` |

---

## 2. Bug 逐个详解

### Bug 1: `RecursionError` — 无限递归

**错误信息**：
```
RecursionError: maximum recursion depth exceeded
  _add_processed_request → ParallelSampleSequenceGroup.add_request → _add_processed_request → ...
```

**原因**：

```
_add_processed_request() 的路由逻辑：
  if params.n > 1 or enable_tree_search:
      → ParallelSampleSequenceGroup.add_request()

ParallelSampleSequenceGroup.add_request() 内部：
  params = original_params.clone()
  params.n = 1
  # 但 enable_tree_search 仍然是 True!
  engine._add_processed_request(params)  ← 又走到 ParallelSample 分支 → 无限递归
```

当 `n=1 + enable_tree_search=True` 时：
1. `_add_processed_request` 看到 `enable_tree_search=True`，走 `ParallelSampleSequenceGroup` 分支
2. `add_request` 克隆 params、设 `n=1`，但旧代码只在 `original_params.n > 1` 时才禁用 tree search
3. 克隆出的 params 仍有 `enable_tree_search=True`
4. 递归调用 `_add_processed_request` → 又走 ParallelSample 分支 → 无限循环

**修复** (`vllm/sequence.py`):
```python
# 修复前：只在 n>1 时禁用
if params.tree_search_params is not None and original_params.n > 1:
    params.tree_search_params.enable_tree_search = False

# 修复后：始终禁用子请求的 tree search（避免递归）
if params.tree_search_params is not None:
    params.tree_search_params.enable_tree_search = False
```

**连带修复** (`vllm/engine/llm_engine.py`):

禁用子请求的 `enable_tree_search` 后，tree decoding 逻辑不能再从子请求的 params 读取 tree search 配置。必须从 **parent group 的原始 params** 读取：

```python
# 修复前：从子请求 params 读取（enable_tree_search=False，永远不触发）
sampling_params = seq_group_metadata.sampling_params
tsp = sampling_params.tree_search_params

# 修复后：从 parent group 的原始 params 读取
current_group = self.seq_id_to_seq_group[request_id]
sampling_params = current_group.assembled_seq_group.sampling_params  # 原始 params
tsp = sampling_params.tree_search_params  # enable_tree_search=True
```

---

### Bug 2: `AssertionError: ppo_mini_batch_size 0 should be larger than 0`

**错误位置**: `fsdp_workers.py:236`

**原因**：

`ppo_mini_batch_size` 的归一化公式：
```python
ppo_mini_batch_size *= n         # 第一步：乘以 n
ppo_mini_batch_size //= n_gpus  # 第二步：除以 GPU 数
```

| 场景 | n | ppo_mini_batch_size | 第一步 | 第二步 (÷4 GPUs) | 结果 |
|------|---|---------------------|--------|-------------------|------|
| 旧配置 n=8 | 8 | 2 | 2×8=16 | 16÷4=4 | OK |
| 新配置 n=1 | 1 | 2 | 2×1=2 | 2÷4=0 | **报错!** |

**修复** (`run_qwen3-8b.sh`):
```bash
# 修复前
actor_rollout_ref.actor.ppo_mini_batch_size=2

# 修复后
actor_rollout_ref.actor.ppo_mini_batch_size=16
# 归一化：16 × 1 ÷ 4 = 4（每GPU），合法
```

---

### Bug 3: `AssertionError: key interaction_kwargs length 24 is not equal to batch size 642`

**错误位置**: `protocol.py:482` (`DataProto.check_consistency`)

**原因**：

在 `vllm_rollout_spmd.py` 中，tree search 把 24 个 prompt 扩展为 ~642 个 leaf response 后：
- `batch` (TensorDict): prompts、responses 等张量已扩展为 642 行 ✓
- `non_tensor_batch`: `interaction_kwargs` 等字段仍是 24 行 ✗

`DataProto.__post_init__` 检查一致性，发现 642 ≠ 24，报错。

**修复** (`vllm_rollout_spmd.py`):
```python
# 在扩展 tensor batch 的同时，也扩展 non_tensor_batch
if len(response) != batch_size:
    prompt_indices_t = torch.tensor(prompt_indices, device=idx.device)
    idx = idx[prompt_indices_t]
    attention_mask = attention_mask[prompt_indices_t]
    position_ids = position_ids[prompt_indices_t]
    # ↓ 新增：用同样的 prompt_indices 扩展 non_tensor_batch
    expanded_ntb = {}
    for k, v in non_tensor_batch.items():
        expanded_ntb[k] = np.array([v[pi] for pi in prompt_indices], dtype=object)
    non_tensor_batch = expanded_ntb
```

`prompt_indices` 是什么？——一个列表，记录每个叶子 response 对应哪个原始 prompt 的下标：
```
prompt_indices = [0,0,...(27个0),1,1,...(27个1),...,23,23,...(27个23)]
# 第0个prompt产生27个叶子，第1个prompt产生27个叶子，...
# 总长度 = 所有叶子数之和 ≈ 642
```

---

### Bug 4: `AssertionError: Conflicting values for meta_info key 'tree_metrics'`

**错误位置**: `protocol.py:959` (`DataProto.concat`)

**原因**：

4 个 worker 各自返回的 `tree_metrics` 略有不同（如 leaf_nodes=642 vs 648），但 `DataProto.concat` 对非 `"metrics"` 的 meta_info key 要求严格相等。

**修复**：
```python
# vllm_rollout_spmd.py — 改用 "metrics" key（concat 有特殊聚合逻辑）
# 修复前
return DataProto(..., meta_info={"tree_metrics": _tree_metrics})
# 修复后
return DataProto(..., meta_info={"metrics": _tree_metrics})

# ray_trainer.py — 从聚合后的 dict-of-lists 中取平均
_agg_metrics = gen_batch_output.meta_info.pop("metrics", {})
for k, v in _agg_metrics.items():
    if k.startswith("tree/"):
        metrics[k] = sum(v) / len(v) if isinstance(v, list) else v
```

---

### Bug 5: `AssertionError: Two tensor dict must have identical batch size. Got torch.Size([96]) and torch.Size([2584])`

**错误位置**: `ray_trainer.py:1085` (`batch.union(gen_batch_output)`)

**原因**：

这是目前 **正在修复** 的问题。训练流程中：

```
batch (96 items)                    ← 原始 prompt batch（含 uid, data_source 等）
    │
    ├── gen_batch = _get_gen_batch(batch)  ← 抽出 input_ids, attention_mask 等给 rollout
    │       │
    │       └── gen_batch_output (2584 items) ← rollout 返回（已经是叶子级别展开的）
    │
    └── batch.repeat(n=1)           ← n=1 时不变，仍是 96 items
         │
         └── batch.union(gen_batch_output) ← 96 vs 2584，维度不匹配!
```

标准 `n>1` 流程中，`batch.repeat(n)` 会把 96 扩展为 96×n 来对齐。但 tree search 的扩展倍数不固定（每个 prompt 叶子数可能不同），所以需要用 `prompt_indices` 做精确映射。

**修复** (`ray_trainer.py`):

```python
if "tree_prompt_indices" in gen_batch_output.non_tensor_batch:
    # local_idx: 每个worker内部的prompt下标 [0~23]
    # num_leaves: 该worker产生了多少叶子（如646），重复存储
    # num_prompts: 该worker处理了多少prompt（如24），重复存储
    local_idx = gen_batch_output.non_tensor_batch["tree_prompt_indices"]
    num_leaves = gen_batch_output.non_tensor_batch["tree_num_leaves"]
    num_prompts = gen_batch_output.non_tensor_batch["tree_num_prompts"]

    # 重建全局下标：每个worker的local_idx要加上前面所有worker的prompt数
    # Worker 0: offset=0,  global_idx = local_idx + 0
    # Worker 1: offset=24, global_idx = local_idx + 24
    # Worker 2: offset=48, global_idx = local_idx + 48
    # Worker 3: offset=72, global_idx = local_idx + 72
    global_idx = reconstruct(local_idx, num_leaves, num_prompts)

    batch = batch[global_idx]  # 96 items → 2584 items（每个叶子对应其prompt的数据）
else:
    batch = batch.repeat(n)    # 标准 n>1 路径
```

**关键辅助字段**（存储在 `non_tensor_batch` 中从 rollout 传回 trainer）：

| 字段 | 长度 | 含义 | 示例 |
|------|------|------|------|
| `tree_prompt_indices` | 叶子数 | 每个叶子对应的 worker-local prompt 下标 | [0,0,...,1,1,...,23,23,...] |
| `tree_num_leaves` | 叶子数 | 该 worker 的总叶子数（重复值，用于分界） | [646,646,...,646] |
| `tree_num_prompts` | 叶子数 | 该 worker 的 prompt 数（重复值，用于计算 offset） | [24,24,...,24] |

---

### Bug 6 (历史): 验证阶段误触发 tree search

**修复** (`vllm_rollout_spmd.py`):
```python
# 验证阶段(do_sample=False)不传tree_search_params
kwargs = {
    "temperature": 0,
    "n": 1,
    "tree_search_params": None,  # ← 关键：显式禁用
}
```

### Bug 7 (历史): `KeyError` — finished 序列被从 `to_be_finished` 移除后仍尝试分叉

**修复** (`llm_engine.py`):
```python
# 在 _process_tree_decoding 中增加检查
if request_id not in current_group.to_be_finished:
    continue  # 已经被 _process_model_outputs 移除了
if seq.is_finished():
    continue  # 序列已经结束（EOS/max_len）
```

### Bug 8 (历史): `IndexError` — logprobs 张量索引越界

logprobs 张量只包含 `do_sample=True` 的序列行，但代码用 metadata 列表的原始下标 `i` 去索引。

**修复**：建立 `logprob_row` 映射表：
```python
logprob_row = {}
row_idx = 0
for i, meta in enumerate(seq_group_metadata_list):
    if meta.do_sample:
        logprob_row[i] = row_idx
        row_idx += 1
# 使用: logprobs[logprob_row[i]] 而不是 logprobs[i]
```

---

## 3. 数据流全景图

```
┌─────────────────────────────────────────────────────────────┐
│                    ray_trainer.py                            │
│                                                             │
│  batch (96 prompts, 含 uid/data_source/reward_model 等)     │
│    │                                                        │
│    ├── gen_batch = _get_gen_batch(batch)                    │
│    │   (抽出 input_ids/attention_mask/position_ids)          │
│    │                                                        │
│    │   gen_batch.repeat(n=1) → 仍是 96                      │
│    │                                                        │
│    ├─── 分发到 4 个 GPU worker ──┐                          │
│    │                              │                         │
└────┼──────────────────────────────┼─────────────────────────┘
     │                              │
     ▼                              ▼
┌─────────────────┐    ┌─────────────────┐
│ Worker 0 (GPU0) │    │ Worker 1 (GPU1) │  ... (×4)
│ 24 prompts      │    │ 24 prompts      │
│                 │    │                 │
│ vLLM generate() │    │ vLLM generate() │
│   ├─ 每个prompt产生│    │                 │
│   │  ≤40个节点   │    │                 │
│   │  (1+3+9+27) │    │                 │
│   │              │    │                 │
│   └─ 收集叶子节点 │    │                 │
│      ≤27个/prompt│    │                 │
│                 │    │                 │
│ 扩展 batch:     │    │                 │
│  24 → ~646 行   │    │                 │
│  (tensor +      │    │                 │
│   non_tensor)   │    │                 │
│                 │    │                 │
│ 返回 DataProto  │    │                 │
│  batch_size=646 │    │                 │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └───── concat ─────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ gen_batch_output (2584 items = 646+646+646+646)             │
│                                                             │
│ 含: prompts, responses, attention_mask, position_ids        │
│     interaction_kwargs (已扩展)                              │
│     tree_prompt_indices, tree_num_leaves, tree_num_prompts  │
│                                                             │
│ ─── 回到 ray_trainer.py ───                                │
│                                                             │
│ batch[global_idx]: 96 items → 2584 items                   │
│ batch.union(gen_batch_output): 合并为完整训练 batch          │
│                                                             │
│ → reward 计算 → log_prob 计算 → GRPO advantage → 策略更新   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 当前状态 & 待验证

- [x] vLLM tree decoding 基础逻辑（entropy 判断 + 分叉）
- [x] 验证阶段不触发 tree search
- [x] 叶子节点收集 & response 提取
- [x] rollout 内部 batch 维度扩展
- [x] non_tensor_batch 维度扩展
- [x] tree metrics 跨 worker 聚合
- [x] ppo_mini_batch_size 适配 n=1
- [ ] **trainer batch 维度对齐（Bug 5，已写代码，待验证）**
- [ ] 端到端训练完整跑通（reward → log_prob → policy update）
- [ ] 移除 `print("entropy:", entropy)` debug 语句
- [ ] 考虑提高 `gpu_memory_utilization`（当前 0.6，可能导致 KV cache preemption）
