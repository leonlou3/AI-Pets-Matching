# P2：匹配算法与 10 用户全流程 Demo

P2 在 P0（Agent 底座）和 P1（宠物孵化）之上，实现了完整的匹配流程：

```text
用户注册（硬条件资料）
→ 访谈训练 + 确认记忆 + 孵化宠物（P1）
→ 硬条件过滤（纯代码，双向性别/年龄/城市）
→ 兼容度初筛（已确认记忆的加权主题重叠）
→ 宠物封闭对谈（有限轮次，只用脱敏记忆）
→ 双向独立裁判（A适合B、B适合A 分别判定）
→ 双方均通过 → 生成推荐（含理由与破冰建议）
```

## 运行 Demo

```bash
source .venv/bin/activate
python -m app.demo
```

Demo 会种入 10 个画像各异的用户（不同性别、城市、年龄偏好、婚育目标），完整走 API 流程，最后打印：

- 匹配漏斗各阶段数量
- 未通过候选对及具体原因
- 推荐结果、推荐理由和破冰建议
- 模型调用次数与 Token 成本

默认使用确定性模拟模型，不消耗 Token。预期结果（模拟模型下应精确复现）：

- 可参与用户 10，评估候选对 45
- 通过硬条件过滤 11，通过兼容度初筛 10
- 完成宠物对谈 10，生成双向推荐 7
- 判负原因包括：性别取向不符、年龄超范围、异地不接受、共同主题不足、婚育目标硬冲突

## 新增 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/users` | 创建用户（硬条件资料） |
| GET | `/v1/users` | 用户列表 |
| POST | `/v1/matching/rounds` | 运行一轮匹配，返回漏斗统计 |
| GET | `/v1/matching/rounds/{id}` | 查看该轮所有候选对、裁判结论、对谈记录和推荐 |
| GET | `/v1/users/{id}/recommendations` | 查看某用户收到的推荐 |

## 匹配算法说明

1. **硬条件过滤**（`app/matching_service.py` 的 `hard_filter`）：纯代码判断，模型不参与。双向检查“寻找的性别”、年龄区间、同城或双方接受异地。
2. **兼容度初筛**（`compatibility`）：对双方已确认记忆的 `memory_key` 做重要度加权重叠计算，低于阈值不进入对谈。阈值在 `.env` 可配置：
   - `MATCHING_MIN_SHARED_MEMORY_KEYS`（默认 2）
   - `MATCHING_MIN_COMPATIBILITY_SCORE`（默认 0.2）
3. **宠物对谈**：每对最多 `MATCHING_DIALOGUE_ROUNDS`（默认 2）轮，每轮双方各发言一次；只读取脱敏记忆；宠物可提前结束对谈。
4. **双向独立裁判**：对 `b_for_a` 和 `a_for_b` 两个方向分别调用裁判 Agent，输出固定结构（结论/置信度/硬冲突/证据/风险/不确定项/破冰建议）。`insufficient_evidence` 不算通过；置信度低于 `MATCHING_JUDGE_MIN_CONFIDENCE`（默认 0.6）不算通过。
5. **推荐**：仅当两个方向都通过才创建，附带可解释理由。

## 给测试者（AI 或人工）的测试清单

### 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 一、自动化测试（必须全部通过）

```bash
pytest -q            # 预期 18 passed
python -m app.evaluation   # 预期 7/7 passed
python -m app.demo         # 预期漏斗数字见上文
```

### 二、单元级验证项

1. **硬过滤**：`tests/test_matching.py::TestHardFilter`
   - 性别寻找不互相匹配 → 拒绝
   - 年龄任一方向超出对方设定 → 拒绝
   - 异城且任一方不接受异地 → 拒绝；双方都接受 → 通过
2. **兼容度**：`TestCompatibility`
   - 无记忆 → 0 分；主题完全相同 → 1 分；部分重叠 → 中间分
3. **模拟裁判**：`TestMockJudge`
   - 婚育目标对立 → fail 且带硬冲突说明
   - 任一方已确认记忆少于 2 条 → insufficient_evidence

### 三、API 级验证项

1. 创建用户成功返回 201；`seeking_min_age > seeking_max_age` 返回 422
2. 无宠物的用户不参与匹配（`eligible_users` 不包含）
3. 匹配轮返回漏斗统计；轮详情包含：
   - 每个候选对的阶段（`hard_filter`/`compatibility`/`dialogue`/`judgement`/`recommended`）和失败原因
   - 双向裁判结论（`a_for_b` 和 `b_for_a` 各一条）
   - 脱敏对谈记录
4. 婚育目标冲突的一对：对谈完成但裁判 fail，无推荐
5. `GET /v1/users/{id}/recommendations` 双方都能查到同一条推荐

### 四、安全与审计验证项

1. 对谈记录和裁判输入不包含手机号、身份证号、邮箱原文（发送前经过脱敏）
2. 每次对谈发言和裁判调用都写入 `agent_runs`（agent_name 为 `pet_dialogue`/`judge`），含 Token 和延迟
3. 硬过滤失败的候选对不产生任何模型调用（成本控制）
4. 只有 `confirmed`/`corrected` 状态的记忆参与匹配；`candidate`/`rejected`/`superseded` 不参与

### 五、边界用例建议（测试者可自行构造）

1. 0 个用户 / 1 个用户时运行匹配轮：应正常返回，`pairs_considered` 为 0
2. 同性取向用户匹配：把双方 `seeking_genders` 设为相同性别验证通过路径
3. 连续运行两轮匹配：第二轮会重新评估（当前版本未去重历史推荐，属已知限制）
4. 修改 `.env` 中的阈值参数后重跑 demo，观察漏斗变化

## 已知限制（后续迭代）

- 已推荐的候选对在下一轮会重复评估，尚无“历史推荐去重”和推荐节流
- 兼容度评分基于主题重叠，尚未引入向量相似度召回
- 匹配轮为同步执行，用户量大时需改为后台异步任务队列（README 第 14 节）
- 模拟模型的裁判规则是确定性的演示逻辑，真实模型下的裁判质量需要单独评测
- `X-Owner-ID` 仍是原型身份边界，匹配 API 未做管理员权限控制
