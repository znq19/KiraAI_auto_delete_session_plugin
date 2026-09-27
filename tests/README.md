# 集成测试（tests/）

`run_tests.py` 在**真实 KiraAI 框架代码**上加载本插件并回归关键场景：使用真实的
`SessionManager`（含框架原生窗口截断）、真实 `LLMRequest` / `Prompt` / 事件类、
真实 `EventBus`（订阅与分发），仅 LLM 调用使用可控的假客户端（可记录调用次数与输入长度）。

## 运行方式

需要一份 KiraAI 框架源码（含 `core/` 目录）：

```bash
# 方式一：指定路径
python tests/run_tests.py --framework /path/to/KiraAI

# 方式二：环境变量
KIRA_FRAMEWORK_PATH=/path/to/KiraAI python tests/run_tests.py
```

脚本也会自动尝试常见相对路径（`../KiraAI`、`../../KiraAI` 等）。找不到框架源码时
安全跳过（退出码 0），不影响仓库其余流程。仓库本身不附带框架源码。

## 覆盖范围（对应 2.1.4 修复）

| 组 | 内容 |
|----|------|
| 静态规范 | manifest 版本/`core_version`/`tags`/图标；schema 无废弃 `enum`、含新文案项；三份默认提示词与 schema 默认值逐字一致 |
| 锚点工具 | `message_fingerprint` / `covered_anchor` / `_locate_anchor` / `_delta_after_anchor`（命中、未命中、锚点丢失等分支） |
| 重开回归 | tokens 触发；工具/推理字段消息的保留与装配；`summarize_mode=off` 不产生摘要不调 LLM |
| 收割（核心） | 真实流程「预压缩 → 回合回写 → 重开」：重开**零同步 LLM 调用**、头部摘要直接来自预压缩结果、后台只补小增量 |
| 滑窗增量 | 框架原生窗口满载滑动时，单轮压缩输入显著小于全量（不再是每轮整段重算） |
| 累计保活 | 头部摘要被原生截断后，store 不丢；下一次重开的摘要仍包含旧摘要内容（旧版此处丢失） |
| 内存桥 | 截断后请求被临时补回摘要头，且**不写记忆** |
| keep 钳制 | tokens 模式下同样保证「保留轮数 < 框架窗口」，钳制后头部摘要存活 |
| 命令 | `/resum` 正常重开（反馈的保留轮数与实际一致）；空会话给出明确提示、不误报成功 |
| 事件清理 | 真实 `EventBus`：清空/删除会话时累计摘要与暂存被同步丢弃；非空写入不受影响；`terminate` 后订阅干净 |
| 生命周期 | `initialize` 后 SessionManager 就绪；重复初始化/终止无残留 |
