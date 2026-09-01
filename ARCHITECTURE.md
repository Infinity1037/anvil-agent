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
    if same call × 3 in one progress phase: inject warning
    if same call × 5 in one progress phase: stop no_progress
    successful write/edit starts a new progress phase
```

停止条件：没有可执行的 tool_calls、输出截断、达到 `max_turns`、连续工具失败、同一进展阶段内相同调用重复五次、Ctrl+C。成功写入或编辑文件后重复计数清零，避免正常的“修改—复测”被误判；没有文件进展时仍会阻止空转。`finish_reason=stop` 但带了工具时仍执行；`finish_reason=tool_calls` 但工具列表为空时当作完成。

## 分层

| 层 | 职责 |
|----|------|
| `cli` / `tui` / `ui` | 解析参数、订阅事件、渲染终端。默认全屏 TUI：思考/工具卡片可原位展开（Ctrl+O），底栏显示估算上下文占用，`/context` 查看明细，`/compact` 手动压缩。回答流式与定稿同一套 Markdown。ask 确认条停靠输入区，Ctrl+E 看全文。`--plain` / `--once` / 非 TTY 走逐行打印，输出统一为 UTF-8，避免 Windows GBK/重定向终端遇到源码 Unicode 字符时崩溃。不执行工具。 |
| `session` | 拥有完整对话历史、checkpoint 与 Skill 激活快照。落在工作区 `.anvil/sessions/`，只按本工作区列出的会话 ID / 唯一前缀恢复。Agent 只跑内层循环。 |
| `agent.loop` | 内层工具循环：无 tool_calls / 截断 / cancel / max_turns / 连续失败 / 无进展时结束。 |
| `agent.permissions` | TUI 默认 `ask`：edit/write/shell 执行前确认；`--once`/`--plain` 默认 `auto`。安全黑名单仍单独生效。 |
| `agent.prompts` | 会话开始时组装一次 system 消息：静态规则在前，环境与工作区 `ANVIL.md`/`AGENTS.md` 在后。项目文件不能覆盖系统规则。 |
| `agent.context` | `prepare(log)` 只生成 view，不改 log；发出去前补齐/丢掉不成对的 tool 消息；token 用最近一次 usage 校准字符比；过大工具结果先落到 `.anvil/tool-output/`，再占位。廉价层仍接近预算时请求一次结构化语义摘要，保留最近完整协议片段。 |
| `skills` | 启动时发现并校验项目内 `.agents/skills/<name>/SKILL.md`，只向 system 暴露有界元数据目录；激活时复核文件哈希并渲染正文。 |
| `llm` | DeepSeek Chat Completions + 流式聚合。退避等待可取消；空流可重试，已有部分响应的中断不重试。畸形 tool_calls 在解析层丢掉或标 `parse_error`，不抛给循环。 |
| `tools` | Schema、本地执行、`ToolResult` + 稳定错误码、Never-throw。 |
| `safety` | 文件/搜索路径约束、密钥文件、shell 明显密钥引用与危险命令黑名单（约定信任，不是 OS 沙箱）。 |

## 工具

`list_dir` `glob` `grep` `read_file` `write_file` `edit_file` `run_shell` `todo`，以及仅在项目存在有效 Skill 时出现的只读 `load_skill`。

- 失败返回 `ToolResult(ok=false, error_code, hint)`，不靠 `"Error:"` 前缀猜；成功的文件修改以 `state_changed` 明确通知循环开启新的进展阶段，不解析展示文本猜副作用。错误码包括 `unknown_tool`、`invalid_json`、`missing_arguments`、`path_escape`、`secret_file`、`dangerous_command`、`command_failed`、`command_timeout`、`stale_read`、`not_unique`、`not_found` 等。
- `read_file` 记下文件摘要；`edit_file` 与覆盖写必须先读且摘要未变，否则 `stale_read`。
- `edit_file`：精确唯一替换；0 次 / 多次匹配返回可恢复错误；成功时附 unified diff。
- `write_file`：默认拒绝覆盖已有文件。
- `read_file`：带行号，大文件强制 offset/limit。
- `glob` / `grep`：跳过密钥文件和越出工作区的文件链接；模板型 `.env.example` 可见。
- `run_shell`：子进程不继承典型凭据环境变量；非零退出是 `command_failed`；超时/取消在 Unix 杀进程组，在 Windows 关闭带 `KILL_ON_JOB_CLOSE` 的 Job Object。大输出不在 shell 层提前丢失。
- 未知参数键会被丢掉，不传给 handler。
- `load_skill` 只返回项目指导文本和资源根路径；frontmatter 的 `allowed-tools` 不参与权限判定，脚本仍通过普通工具和审批执行。

## Skill 生命周期

项目 Skill 采用三段式渐进披露：会话启动只扫描直接子目录并读取 `name` / `description`；任务匹配后，模型调用 `load_skill` 获取完整正文；正文引用的资源再由现有文件或 shell 工具按需处理。目录、数量、文件大小、UTF-8、YAML、名称和工作区边界均在本地校验，发现后保存文件 SHA-256，加载时若内容变化则失败，避免检查后替换。

`disable-model-invocation: true` 的 Skill 不进入模型目录，`load_skill` 也拒绝模型调用，但用户仍可用斜杠命令激活。Flow 类型不解析、不执行；`allowed-tools` 仅被当作不生效的元数据。

手动 `/skill:<name> [任务]` 与模型调用共用相同的渲染、预算和持久化规则。激活来源消息与 `skill_activation` 记录在一次 flush 中写入 JSONL，记录包含来源下标与哈希。上下文 checkpoint 若覆盖了来源消息，构建模型 view 时会重新附加原始激活正文；resume 重新校验来源哈希后才恢复。Skill 始终是低优先级项目指导，不授予权限，也不会绕过工具安全边界。

## DeepSeek

默认 `deepseek-v4-flash`，思考模式开启，`reasoning_effort=max`。Anvil 为日常 Coding Agent 工作流采用 200,000-token 有效窗口、160,000-token 安全模型视图和 32,000-token 单轮输出上限，为工具 schema 与估算误差保留余量；底栏按有效窗口显示占用，`/context` 分列各层限制。语义 checkpoint 在有效窗口的 85% 与安全视图上限中较早的位置触发，默认约 160,000 tokens；更早先折叠旧工具结果。带 `tools` 时每一轮 assistant 必须回传 `reasoning_content`，否则 HTTP 400。会话内 `/effort` 只改下一轮请求，不改历史。400/401 不重试；429/5xx 有限退避。

## 上下文 checkpoint

上下文采用“廉价层优先，语义摘要兜底”：工具输出进入历史前先完整落盘，模型预览严格遵守字符上限；接近预算后依次折叠旧工具结果。若保留历史语义的视图仍超过软阈值，Agent 用同一个 LLM 发起一次 `tools=[]`、思考关闭、输出受限的摘要请求，固定保留目标、约束、决定、文件变化、测试结果和剩余工作。

checkpoint 只保存摘要、覆盖到的原始消息下标、源历史 SHA-256、摘要 usage 与模型名。模型视图由“原 system + 历史摘要 + 未覆盖尾部”临时重建，原始消息不删除。会话恢复时重新计算源历史哈希，只有匹配的最新 checkpoint 才会复用。后续摘要只处理上个覆盖点之后的新增历史。

自动压缩与 `/compact [关注点]` 共用同一个原子入口：摘要禁用工具与思考并限制输出；历史与关注点作为编码后的不可信数据进入提示，关注点只能调整强调内容，不能删除固定保留项。手动压缩仅在 Agent 空闲边界启动；摘要无效、截断、返回工具调用、被取消或 checkpoint 写盘失败时，内存和原 checkpoint 均不切换。`/context` 展示相对于 `ANVIL_CONTEXT_BUDGET` 的模型视图估算、剩余预算与覆盖范围；API 返回 usage 后校准估算，因此界面使用 `≈` 而不冒充服务端精确 tokenizer。

为了避免极低预算导致“摘要后读取文件、下一步又摘要掉精确内容”的抖动，同一用户轮次最多尝试一次自动语义摘要；之后若仍需缩减，只使用确定性视图裁剪。最终视图仍超过硬预算时以 `context_overflow` 停止，不向 API 发送已知超限请求。

## 明确不做

多 agent、MCP、插件市场、子代理、会话树、向量库、把现成 agent 产品当运行时依赖。
