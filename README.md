# Anvil

Anvil 是一个运行在本机工作区上的编程智能体。模型只负责决定下一步；读文件、改文件、搜代码、跑命令都在本地执行，结果再写回对话，循环直到任务完成。

默认对接 [DeepSeek API](https://api-docs.deepseek.com/) 的 OpenAI 兼容 Chat Completions（`deepseek-v4-flash`），使用原生 tool calling。不依赖 LangChain、LlamaIndex、Agents SDK 等 agent 框架，也不使用服务端 Code Interpreter / Files API。

## 它做什么

- 在指定工作区里自主探索并修改代码
- 用精确字符串替换编辑已有文件（匹配必须唯一）
- 执行测试和命令，根据输出继续改
- 截断过大的工具结果，并在上下文接近预算时压缩早期输出
- 把路径限制在工作区内，拒绝明显危险的命令

```
anvil --workspace examples/broken_ledger "Make the tests pass. Run python -m unittest -v"
```

## 安装

需要 Python 3.10+（推荐 3.11）。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`。不要把真实 key 提交到 git。

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## 配置

| 变量 | 默认 | 含义 |
|------|------|------|
| `DEEPSEEK_API_KEY` | （必填） | DeepSeek API key |
| `ANVIL_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容网关 |
| `ANVIL_MODEL` | `deepseek-v4-flash` | 模型 id，可改为 `deepseek-v4-pro` |
| `ANVIL_THINKING` | `1` | 是否开启思考模式 |
| `ANVIL_REASONING_EFFORT` | `high` | `low` / `high` / `max` |
| `ANVIL_MAX_TURNS` | `40` | 最大循环轮次 |
| `ANVIL_CONTEXT_BUDGET` | `100000` | 触发压缩的估算 token 预算 |

命令行：`anvil --model deepseek-v4-flash --effort high --workspace . "your task"`。不带任务参数则进入 REPL（`/reset`、`/quit`）。

工作区根目录若存在 `ANVIL.md` 或 `AGENTS.md`，会注入 system prompt。

## 循环如何运转

```
messages = [system, user_task]
loop:
    调模型（tools + 原生 function calling）
    若返回 tool_calls → 本地执行 → 把结果作为 tool 消息追加
    若不再调工具 → 结束
    达到 max_turns / 连续工具失败 / Ctrl+C → 停止
```

工具：`list_dir`、`glob`、`grep`、`read_file`、`write_file`、`edit_file`、`run_shell`、`todo`。只读工具可以并行；写入和 shell 串行。

### 为什么必须回传 `reasoning_content`

DeepSeek 思考模式默认开启。官方文档写明：只要请求里带了 `tools`，后续每一轮 assistant 消息都必须带上上一轮的 `reasoning_content`，否则接口返回 400。Anvil 把该字段当作一等公民写进历史，而不是只保存 `content`。

## 开发

```powershell
pytest
```

单元测试使用 ScriptedLLM，不访问网络、不消耗额度。`examples/broken_ledger` 是一个测试会失败的小账本，用来演示真实修 bug 流程。

## 设计取舍

- **原生 tool calling，而不是自己解析 XML。** 参数是 JSON Schema，少一次脆弱的文本解析。
- **专用文件工具 + shell，而不是只给一个 bash。** `edit_file` 能校验替换次数；搜索工具避免把整仓塞进上下文；Windows 上也不依赖 Unix 工具链。
- **精确替换而不是整文件覆盖。** 匹配 0 次或多次都返回可操作的错误，让模型重读后再改。
- **上下文廉价优先压缩。** 先截断单次工具输出，再折叠早期结果。窗口即使是 1M，测试日志也能把它撑满。
- **工具永不把进程打崩。** 异常变成带提示的字符串，交给模型下一轮自愈。

会话记录写在工作区 `.anvil/transcripts/`（已 gitignore）。
