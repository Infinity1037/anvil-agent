# Anvil

Anvil 是一个运行在本机工作区上的编程智能体。模型只负责决定下一步；读文件、改文件、搜代码、跑命令都在本地执行，结果再写回对话，循环直到任务完成。

默认对接 [DeepSeek API](https://api-docs.deepseek.com/) 的 OpenAI 兼容 Chat Completions（`deepseek-v4-flash`），使用原生 tool calling。不依赖 LangChain、LlamaIndex、Agents SDK 等 agent 框架，也不使用服务端 Code Interpreter / Files API。

## 它做什么

- 在指定工作区里自主探索并修改代码
- 用精确字符串替换编辑已有文件（匹配必须唯一，成功后返回 unified diff）
- `write_file` 默认拒绝覆盖已有文件，避免整文件误伤
- 执行测试和命令，根据输出继续改；同一调用第三次提醒换策略，第五次停止
- 截断过大的工具结果；压缩只作用于发给模型的视图，JSONL 保留完整轨迹
- 文件与搜索工具限制在工作区内并跳过密钥文件；shell 不继承凭据环境变量，并拒绝明显危险或直接引用密钥文件的命令

默认是**全屏终端会话**（做完一件事还可以追问）。思考链和工具输出默认只显示预览，**Ctrl+O 在原来的卡片上展开/收起**，不会在输入框下面再打印一份。也可以只跑一次就退出。

```
# 进入对话（推荐）
anvil --workspace examples/broken_ledger

# 先做一件事，做完后仍停在聊天里，可以继续说「再加测试」「解释你改了什么」
anvil --workspace examples/broken_ledger "Make the tests pass. Run python -m unittest -v"

# 脚本/录屏：做完一件事立刻退出
anvil --once --workspace examples/broken_ledger "Make the tests pass"
```

会话里可以输入：

- 普通中文/英文，当作下一轮任务（历史、工作区、todo 都保留）
- `Ctrl+O` 在思考链 / 工具卡片**原位**展开或收起（再按一次收起）
- `Enter` 发送，`Shift+Enter` 换行
- `/` 弹出命令补全（按前缀过滤：help / status / effort / perm / clear / resume / expand / exit）
- `@` 弹出工作区文件补全
- `?` 或 `F1` 打开键位说明（输入框为空时）
- `/status` 看模型、思考强度、会话与 token
- `/effort` 弹出思考强度选择（off / low / high / max）；当前档位显示在底栏
- `/perm` 或 `Shift+Tab` 切换权限：`ask` 在改文件和跑命令前确认，`auto` 不问（`--once` / `--plain` 默认 auto）。确认条只展示摘要，Ctrl+E 看全文，Tab 可写拒绝原因
- `/clear` 开始新会话（`/new` 仍可用）
- `/resume` 恢复本工作区历史会话；启动可用 `anvil --continue`
- `/exit` 或 `Ctrl+Q` 退出（`/quit` 仍可用）

全屏会话结束后，终端滚动区不会留下本次对话；记录在工作区 `.anvil/sessions/`。若需要旧的逐行打印（Ctrl+O 会在输入框下面重打全文），加 `--plain`。

## 安装

需要 Python 3.11+。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`。不要把真实 key 提交到 git。

Linux / macOS：

```bash
python3.11 -m venv .venv
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
| `ANVIL_REASONING_EFFORT` | `max` | `low` / `high` / `max`；复杂 Agent 默认最强思考 |
| `ANVIL_MAX_TOKENS` | `256000` | 单次请求输出上限（含思考链） |
| `ANVIL_REQUEST_TIMEOUT` | `300` | 流式空闲超时（秒） |
| `ANVIL_MAX_TURNS` | `40` | 最大循环轮次 |
| `ANVIL_CONTEXT_BUDGET` | `100000` | 触发压缩的估算 token 预算 |

命令行：`anvil --workspace .` 进入全屏对话；后面可以跟一句首条任务。`--once` 表示只跑这一次。`--plain` 关闭全屏 TUI。`--effort low|high|max` 覆盖启动时的思考强度。`--continue` 恢复本工作区最近一次会话。`/status` `/effort` `/clear` `/resume` `/exit` 是会话内命令。

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

更完整的分层说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 设计取舍

- **原生 tool calling，而不是自己解析 XML。** 参数是 JSON Schema，少一次脆弱的文本解析。
- **专用文件工具 + shell，而不是只给一个 bash。** `edit_file` 能校验替换次数；搜索工具避免把整仓塞进上下文；Windows 上也不依赖 Unix 工具链。
- **精确替换而不是整文件覆盖。** 匹配 0 次或多次都返回可操作的错误，让模型重读后再改；写文件默认不能覆盖。
- **上下文廉价优先压缩。** 先截断单次工具输出，再折叠早期结果。发给模型的是视图，磁盘上的 transcript 仍是全文。
- **工具永不把进程打崩。** 异常变成带提示的字符串，交给模型下一轮自愈。
- **命令失败不是成功。** 非零退出带 `command_failed` 返回；超时或取消会终止 shell 进程树。完整大输出交给上下文层落盘后再裁剪模型视图。
- **无进展检测。** 连续三次完全相同的工具调用会在结果里插入换策略提醒，避免空转。

会话记录写在工作区 `.anvil/sessions/`（已 gitignore）。
