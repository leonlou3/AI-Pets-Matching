# P0 Agent 基础设施

这一阶段提供可运行的 Agent 后端底座，不包含完整的宠物孵化和匹配算法。

## 已实现

- FastAPI 统一后端入口与自动 API 文档
- SQLite 开箱即用，PostgreSQL 生产兼容配置
- OpenAI-compatible 模型网关、超时、有限重试和模拟模型
- Pydantic 结构化输出校验
- 对话、候选记忆、确认状态和 Agent 运行指标持久化
- 手机号、身份证号和邮箱在发送给模型前脱敏
- Prompt 独立版本管理
- 模型、Prompt、Token、延迟、费用和错误码记录
- 不调用真实模型的端到端自动测试

## 本地运行

需要 Python 3.12 或以上版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

默认 `MODEL_PROVIDER=mock`，不会发送外部请求或产生 Token 费用。打开：

- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

运行测试：

```bash
pytest
```

## 体验完整链路

创建访谈：

```bash
curl -X POST http://127.0.0.1:8000/v1/interviews \
  -H 'Content-Type: application/json' \
  -d '{"owner_id":"demo-user"}'
```

复制响应中的 `id`，发送消息：

```bash
curl -X POST http://127.0.0.1:8000/v1/interviews/<id>/messages \
  -H 'Content-Type: application/json' \
  -H 'X-Owner-ID: demo-user' \
  -d '{"content":"我不接受异地恋"}'
```

响应会包含访谈回复、状态为 `candidate` 的候选记忆，以及本次调用指标。候选记忆需要用户通过 `PATCH /v1/memories/{id}` 确认后才成为正式记忆。P1 的完整孵化流程见 [P0_P1_OVERVIEW.md](P0_P1_OVERVIEW.md)。

## 接入真实模型

在 `.env` 中设置：

```dotenv
MODEL_PROVIDER=openai_compatible
MODEL_BASE_URL=https://模型供应商的兼容接口/v1
MODEL_API_KEY=服务端密钥
MODEL_NAME=模型名称
```

密钥不得提交到 Git，也不得放入网页或小程序代码。

## PostgreSQL

仓库提供 `compose.yaml` 作为本地 PostgreSQL 示例。启动数据库后，将配置改为：

```dotenv
DATABASE_URL=postgresql+asyncpg://ai_pets:local_only_change_me@localhost:5432/ai_pets
```

## 当前边界

- `X-Owner-ID` 只是方便 P0 演示的身份边界，不是真实认证。对外开放前必须替换为经过验证的微信登录身份。
- P0 为便于演示自动建表；正式部署前应增加版本化数据库迁移。
- 当前脱敏覆盖常见手机号、身份证号和邮箱；生产环境还需要供应商内容安全服务、访问审计和更完整的中文实体识别。
- P1 已在底座上增加动态访谈、冲突记忆、用户确认、宠物画像和反向验证；匹配与双宠物对谈仍未实现。
