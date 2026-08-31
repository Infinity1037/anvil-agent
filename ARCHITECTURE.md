# Anvil 架构

Anvil 是一个单智能体、线性历史的本地 coding agent。模型只产出决策；文件和命令在本机执行，结果写回对话后再进入下一轮。

## 循环

```
messages = [system, user]
loop:
    view = compact(messages)          # 只改发给模型的视图，完整历史保留
    response = llm(view, tools)
    append assistant message          # 含 reasoning_content（DeepSeek 思考+工具硬性要求）
    if no executable tool_calls: stop # 不盲信 finish_reason；仅 length 视为截断
    results = execute(tool_calls)     # ToolResult；只读可并行，写入/shell 串行
    append tool messages
    if same call × 3: inject progress warning
    if same call × 5: stop no_progress
```

停止条件：没有可执行的 tool_calls、输出截断、达到 `max_turns`、连续工具失败、同一调用重复五次、Ctrl+C。`finish_reason=stop` 但带了工具时仍执行；`finish_reason=tool_calls` 但工具列表为空时当作完成。

## 分层

| 层 | 职责 |
|----|------|
| `cli` / `tui` / `ui` | 解析参数、订阅事件、渲染终端。默认全屏 TUI：思考/工具卡片可原位展开（Ctrl+O）。回答流式与定稿同一套 Markdown。ask 确认条停靠输入区，Ctrl+E 看全文。`--plain` / `--once` / 非 TTY 走逐行打印。不执行工具。 |
| `session` | 拥有完整对话历史。落在工作区 `.anvil/sessions/`，可 `/resume` 或 `anvil --continue` 读回。Agent 只跑内层循环。 |
| `agent.loop` | 内层工具循环：无 tool_calls / 截断 / cancel / max_turns / 连续失败 / 无进展时结束。 |
| `agent.permissions` | TUI 默认 `ask`：edit/write/shell 执行前确认；`--once`/`--plain` 默认 `auto`。安全黑名单仍单独生效。 |
| `agent.prompts` | 会话开始时组装一次 system 消息：静态规则在前，环境与工作区 `ANVIL.md`/`AGENTS.md` 在后。项目文件不能覆盖系统规则。 |
| `agent.context` | `prepare(log)` 只生成 view，不改 log；发出去前补齐/丢掉不成对的 tool 消息；token 用最近一次 usage 校准字符比；过大工具结果先落到 `.anvil/tool-output/`，再占位、丢中间轮。 |
| `llm` | DeepSeek Chat Completions + 流式聚合。畸形 tool_calls 在解析层丢掉或标 `parse_error`，不抛给循环。 |
| `tools` | Schema、本地执行、`ToolResult` + 稳定错误码、Never-throw。 |
| `safety` | 工作区路径约束、密钥文件、危险命令黑名单（约定信任，不是 OS 沙箱）。 |

## 工具

`list_dir` `glob` `grep` `read_file` `write_file` `edit_file` `run_shell` `todo`

- 失败返回 `ToolResult(ok=false, error_code, hint)`，不靠 `"Error:"` 前缀猜。错误码包括 `unknown_tool`、`invalid_json`、`missing_arguments`、`path_escape`、`secret_file`、`dangerous_command`、`stale_read`、`not_unique`、`not_found` 等。
- `read_file` 记下文件摘要；`edit_file` 与覆盖写必须先读且摘要未变，否则 `stale_read`。
- `edit_file`：精确唯一替换；0 次 / 多次匹配返回可恢复错误；成功时附 unified diff。
- `write_file`：默认拒绝覆盖已有文件。
- `read_file`：带行号，大文件强制 offset/limit。
- 未知参数键会被丢掉，不传给 handler。

## DeepSeek

默认 `deepseek-v4-flash`，思考模式开启，`reasoning_effort=max`，`max_tokens=256000`。带 `tools` 时每一轮 assistant 必须回传 `reasoning_content`，否则 HTTP 400。会话内 `/effort` 只改下一轮请求，不改历史。400/401 不重试；429/5xx 有限退避。

## 明确不做

多 agent、MCP、插件市场、子代理、会话树、向量库、把现成 agent 产品当运行时依赖。
