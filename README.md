# Anvil

Anvil 是一个运行在本机工作区上的编程智能体。模型只负责决定下一步；读文件、改文件、搜代码、跑命令都在本地执行，结果再写回对话，循环直到任务完成。

默认对接 [DeepSeek API](https://api-docs.deepseek.com/) 的 OpenAI 兼容 Chat Completions（`deepseek-v4-flash`），使用原生 tool calling。不依赖 LangChain、LlamaIndex、Agents SDK 等 agent 框架，也不使用服务端 Code Interpreter / Files API。

## 它做什么

- 在指定工作区里自主探索并修改代码
- 用精确字符串替换编辑已有文件（匹配必须唯一，成功后返回 unified diff）
- `write_file` 默认拒绝覆盖已有文件，避免整文件误伤
- 执行测试和命令，根据输出继续改；同一调用第三次提醒换策略，第五次停止
- 超大工具结果落盘并严格裁剪预览；廉价层仍超预算时生成语义 checkpoint，压缩只作用于模型视图，JSONL 保留完整轨迹
- 按开放 `SKILL.md` 格式发现项目级 Skill：名称和描述常驻、正文按需加载，激活状态可跨 checkpoint 与 resume 保留
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
- `/` 弹出命令补全（按前缀过滤：help / status / context / compact / skills / effort / perm / clear / resume / expand / exit）
- `@` 弹出工作区文件补全
- `?` 或 `F1` 打开键位说明（输入框为空时）
- `/status` 看模型、思考强度、会话与 token；底栏持续显示估算的上下文预算占用
- `/context` 查看当前模型视图、预算、完整/活动消息数以及 checkpoint 状态
- `/compact [关注点]` 在空闲时手动生成 checkpoint；例如 `/compact 保留测试结果和剩余任务`
- `/skills` 查看当前项目发现的 Skill；`/skill:<名称> [任务]` 显式激活，也可由模型按描述自动选择
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
| `ANVIL_MAX_TOKENS` | `32000` | 单次请求输出上限（含思考链） |
| `ANVIL_CONTEXT_WINDOW` | `200000` | Agent 有效上下文窗口；底栏与 `/context` 以此显示用量 |
| `ANVIL_REQUEST_TIMEOUT` | `300` | 流式空闲超时（秒） |
| `ANVIL_MAX_TURNS` | `40` | 最大循环轮次 |
| `ANVIL_CONTEXT_BUDGET` | `160000` | Agent 安全模型视图上限；为输出和协议开销预留空间 |

上下文窗口、Agent 视图预算和单轮输出是三个独立概念。配置必须满足 `ANVIL_CONTEXT_BUDGET + ANVIL_MAX_TOKENS <= ANVIL_CONTEXT_WINDOW`；底栏按有效窗口显示占用，`/context` 会同时列出三者和自动压缩边界。默认在有效窗口的 85% 与安全视图上限两者中较早的位置生成语义 checkpoint。

命令行：`anvil --workspace .` 进入全屏对话；后面可以跟一句首条任务。`--once` 表示只跑这一次。`--plain` 关闭全屏 TUI。`--effort low|high|max` 覆盖启动时的思考强度。`--continue` 恢复本工作区最近一次会话。`/status` `/context` `/compact` `/skills` `/skill:<名称>` `/effort` `/clear` `/resume` `/exit` 是会话内命令。

工作区根目录若存在 `ANVIL.md` 或 `AGENTS.md`，会注入 system prompt。

### 项目 Skill

Skill 放在工作区 `.agents/skills/<name>/SKILL.md`，使用 YAML frontmatter 和 Markdown 正文：

```markdown
---
name: test-first
description: Use when adding or changing tested behavior.
---

Run the relevant tests before and after the change. Task: $ARGUMENTS
```

会话开始时只把有效 Skill 的名称和描述放进目录；模型匹配到任务后通过只读 `load_skill` 加载正文，引用的 `references/`、`scripts/` 等资源仍用普通文件工具按需读取。Skill 只提供项目指导：不能覆盖系统规则或用户要求，`allowed-tools` 不会放宽权限，脚本也不会自动执行。Skill 文件在发现后若被修改，会拒绝加载并要求新会话刷新，避免检查与使用之间被替换。

需要只允许用户手动触发时，可在 frontmatter 写 `disable-model-invocation: true`；它仍会出现在 `/skills`，但不会进入模型目录，只能通过 `/skill:<名称>` 激活。`type: flow` 不在本项目范围内，会被明确忽略。

## 循环如何运转

```
messages = [system, user_task]
loop:
    调模型（tools + 原生 function calling）
    若返回 tool_calls → 本地执行 → 把结果作为 tool 消息追加
    若不再调工具 → 结束
    达到 max_turns / 连续工具失败 / Ctrl+C → 停止
```

基础工具：`list_dir`、`glob`、`grep`、`read_file`、`write_file`、`edit_file`、`run_shell`、`todo`。项目存在有效 Skill 时增加只读 `load_skill`。只读工具可以并行；写入和 shell 串行。

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
- **上下文廉价优先压缩。** 先将超大工具输出落盘并保留严格限长的首尾预览，再折叠旧结果；仍接近预算才用一次无工具模型调用生成结构化语义 checkpoint。checkpoint 带覆盖范围与源历史哈希并写入 JSONL，恢复会话时校验后复用；完整 transcript 始终保留。同一用户轮次最多自动摘要一次，避免低预算下反复摘要、重读文件。也可用 `/compact [关注点]` 在任务边界手动触发；`/context` 与底栏让压缩前后占用可观察。
- **不发送注定超限的请求。** 压缩后会再次检查预算；若 system 与当前用户输入本身已经不可压缩，明确停止为 `context_overflow`，而不是等待 API 拒绝或静默截断任务。
- **Skill 渐进披露而不是全量塞入提示词。** 启动只暴露有预算上限的元数据目录；激活正文与来源哈希写入会话。checkpoint 覆盖激活消息后会精确重挂正文，resume 会校验来源后恢复，避免摘要丢掉项目约束。
- **工具永不把进程打崩。** 异常变成带提示的字符串，交给模型下一轮自愈。
- **命令失败不是成功。** 非零退出带 `command_failed` 返回；超时或取消会终止 shell 进程树。完整大输出交给上下文层落盘后再裁剪模型视图。
- **无进展检测。** 连续三次完全相同的工具调用会在结果里插入换策略提醒，避免空转。

会话记录写在工作区 `.anvil/sessions/`（已 gitignore）。
