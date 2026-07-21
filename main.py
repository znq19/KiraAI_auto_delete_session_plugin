import asyncio
import time
from collections import defaultdict
from typing import Any, List, Dict, Optional

from core.plugin import BasePlugin, logger, on, Priority
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent
from core.chat import MessageChain
from core.chat.message_elements import Text
from core.provider.llm_model import LLMRequest
from core.agent.message import OpenAIMessage

from .summarizer import (
    DEFAULT_SUMMARIZE_PROMPT,
    build_summary_chunk,
    is_summary_chunk,
    summarize_history,
)


class AutoDeleteSessionPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.last_check = defaultdict(float)
        self.max_tokens = int(cfg.get("max_tokens", 5000))
        self.chars_per_token = float(cfg.get("chars_per_token", 2.0))
        self.check_interval = int(cfg.get("check_interval_seconds", 60))
        self.keep_recent_turns = int(cfg.get("keep_recent_turns", 3))

        # 重开前摘要
        sec = cfg.get("section_summarize", {})
        if not isinstance(sec, dict):
            sec = {}
        self.summarize_mode = str(sec.get("summarize_mode", "sync") or "sync").lower()
        if self.summarize_mode not in ("off", "sync", "async"):
            self.summarize_mode = "sync"
        self.summarize_model = str(sec.get("summarize_model", "") or "")
        self.summarize_timeout_sec = float(sec.get("summarize_timeout_sec", 5.0) or 5.0)
        # 注意：0 是合法值（表示无上限），不能用 `or 默认值` 读取
        _mic = sec.get("summarize_max_input_chars", 6000)
        self.summarize_max_input_chars = int(_mic if _mic is not None else 6000)
        _moc = sec.get("summarize_max_output_chars", 3000)
        self.summarize_max_output_chars = int(_moc if _moc is not None else 3000)
        self.summarize_prompt_template = str(
            sec.get("summarize_prompt_template", "") or ""
        ) or DEFAULT_SUMMARIZE_PROMPT
        self.enable_summary_logging = bool(sec.get("enable_summary_logging", False))

        # 手动压缩重开命令（对齐 reboot_all / 自定义命令：list 关键词 + 白名单 + 整句匹配）
        cmd = cfg.get("section_command", {})
        if not isinstance(cmd, dict):
            cmd = {}
        self.enable_reset_command = bool(cmd.get("enable_reset_command", False))
        cmds = cmd.get("reset_commands", ["/resum"])
        if isinstance(cmds, str):
            cmds = [cmds]
        self.reset_commands = [str(c).strip() for c in (cmds or []) if str(c).strip()]
        # 兼容旧版 trigger_keywords（逗号分隔字符串）：并入命令列表
        legacy_kw = str(sec.get("trigger_keywords", "") or "").strip()
        if legacy_kw:
            for kw in legacy_kw.split(","):
                kw = kw.strip()
                if kw and kw not in self.reset_commands:
                    self.reset_commands.append(kw)
        if not self.reset_commands:
            self.reset_commands = ["/resum"]
        self.reset_enable_permission = bool(cmd.get("reset_enable_permission", False))
        rau = cmd.get("reset_allowed_users", [])
        if isinstance(rau, str):
            rau = [x.strip() for x in rau.split(",") if x.strip()]
        self.reset_allowed_users = [str(u).strip() for u in (rau or []) if str(u).strip()]
        self.reset_success_message = str(
            cmd.get(
                "reset_success_message",
                "✅ 本会话已压缩重开，保留最近 {keep} 轮{summary}，我们可以重新开始了！",
            )
            or "✅ 本会话已压缩重开，保留最近 {keep} 轮{summary}，我们可以重新开始了！"
        )
        self.reset_permission_denied_message = str(
            cmd.get(
                "reset_permission_denied_message",
                "❌ 权限不足：您没有压缩重开会话的权限",
            )
            or "❌ 权限不足：您没有压缩重开会话的权限"
        )
        self.reset_error_message = str(
            cmd.get("reset_error_message", "❌ 压缩重开失败: {error}")
            or "❌ 压缩重开失败: {error}"
        )

        self.session_mgr = None
        self._dynamic_keep_turns: Dict[str, int] = {}
        self._last_reset_time: Dict[str, float] = {}
        self._summary_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        if hasattr(self.ctx, 'session_mgr'):
            self.session_mgr = self.ctx.session_mgr
        else:
            self.session_mgr = self._find_session_manager()

        if self.session_mgr is None:
            logger.error("AutoDelete: 无法找到 SessionManager，插件无法工作")
            return

        required = ['fetch_memory', 'read_memory', 'write_memory', 'delete_session', 'get_session_info']
        missing = [m for m in required if not hasattr(self.session_mgr, m)]
        if missing:
            logger.error(f"AutoDelete: SessionManager 缺少方法: {missing}")
            self.session_mgr = None
            return

        logger.info(
            f"AutoDeletePlugin 初始化完成: max_tokens={self.max_tokens}, "
            f"chars_per_token={self.chars_per_token}, check_interval={self.check_interval}s, "
            f"keep_recent_turns={self.keep_recent_turns}, summarize={self.summarize_mode}"
        )

    def _find_session_manager(self):
        candidates = ['session_mgr', 'session_manager', 'mem_mgr', 'memory_manager']
        for name in candidates:
            if hasattr(self.ctx, name):
                obj = getattr(self.ctx, name)
                if obj and hasattr(obj, 'delete_session'):
                    return obj
        return None

    async def terminate(self):
        self.last_check.clear()
        self._dynamic_keep_turns.clear()
        self._last_reset_time.clear()
        for t in list(self._summary_tasks.values()):
            if t and not t.done():
                t.cancel()
        self._summary_tasks.clear()
        logger.info("AutoDeletePlugin 已终止")

    def count_tokens(self, text: Any) -> int:
        if not isinstance(text, str):
            text = str(text)
        return max(1, int(len(text) / self.chars_per_token) + 1)

    def _clean_and_chunk(self, flat: List[dict]) -> List[List[dict]]:
        """清理并分割消息：丢弃开头的非user消息，然后按user边界分割"""
        # 找到第一个 role == "user" 的位置
        start_idx = 0
        for i, msg in enumerate(flat):
            if msg.get("role") == "user":
                start_idx = i
                break
        else:
            logger.warning("历史中没有 user 消息，无法分割 chunk")
            return []

        # 从第一个 user 开始切片
        cleaned = flat[start_idx:]

        chunks = []
        cur = []
        for msg in cleaned:
            if msg.get("role") == "user":
                if cur:
                    chunks.append(cur)
                cur = [msg]
            else:
                cur.append(msg)
        if cur:
            chunks.append(cur)
        return chunks

    def _extract_recent_chunks(self, chunks: List[List[dict]], turns: int) -> List[List[dict]]:
        if turns <= 0 or not chunks:
            return []
        return chunks[-turns:] if len(chunks) >= turns else chunks[:]

    def _flatten_chunks(self, chunks: List[List[dict]]) -> List[dict]:
        flat = []
        for c in chunks:
            flat.extend(c)
        return flat

    def _replace_request_messages(self, req: LLMRequest, new_history: List[dict]):
        system_msgs = []
        for m in req.messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "system":
                if isinstance(m, dict):
                    system_msgs.append(OpenAIMessage(**m))
                else:
                    system_msgs.append(m)

        new_objs = [OpenAIMessage(**msg) for msg in new_history]

        current_user = None
        for m in reversed(req.messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "user":
                if isinstance(m, dict):
                    current_user = OpenAIMessage(**m)
                else:
                    current_user = m
                break

        new_msgs = system_msgs + new_objs
        if current_user:
            new_msgs.append(current_user)
        req.messages = new_msgs
        logger.info(f"本次请求 messages 已替换: 共 {len(new_msgs)} 条")

    # ── 重开前摘要 ─────────────────────────────────────────────

    def _dropped_flat(
        self, old_chunks: List[List[dict]], keep_turns: int
    ) -> List[dict]:
        """keep 轮之外将被丢弃的消息（摘要输入）。"""
        if not old_chunks:
            return []
        if keep_turns <= 0 or len(old_chunks) <= keep_turns:
            return []
        return self._flatten_chunks(old_chunks[:-keep_turns])

    async def _summarize_dropped(self, sid: str, dropped: List[dict]) -> Optional[str]:
        return await summarize_history(
            self.ctx,
            sid,
            dropped,
            model_id=self.summarize_model,
            prompt_template=self.summarize_prompt_template,
            timeout_sec=self.summarize_timeout_sec,
            max_input_chars=self.summarize_max_input_chars,
            max_output_chars=self.summarize_max_output_chars,
            logger=logger,
            enable_detail_log=self.enable_summary_logging,
        )

    def _schedule_async_summary(self, sid: str, dropped: List[dict]):
        """async 模式：重开先行，摘要后补写进新记忆最前。"""
        old_task = self._summary_tasks.get(sid)
        if old_task and not old_task.done():
            old_task.cancel()

        async def _run():
            try:
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] [async] 后台任务开始，sid={sid}")
                summary = await self._summarize_dropped(sid, dropped)
                if not summary or self.session_mgr is None:
                    if self.enable_summary_logging:
                        logger.info(f"[摘要调试] [async] 摘要生成失败或 session_mgr 不可用")
                    return
                chunks = self.session_mgr.read_memory(sid) or []
                # 会话又被删/重开过且已有摘要，或已注入过 → 放弃补写
                if chunks and is_summary_chunk(chunks[0]):
                    if self.enable_summary_logging:
                        logger.info(f"[摘要调试] [async] 会话已有摘要，放弃补写")
                    return
                new_chunks = [build_summary_chunk(summary)] + list(chunks)
                self.session_mgr.write_memory(sid, new_chunks)
                logger.info(f"✅ [async] 摘要已补写入会话 {sid} 首部")
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] [async] 补写的摘要内容:\n{summary}")
            except asyncio.CancelledError:
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] [async] 后台任务被取消，sid={sid}")
                pass
            except Exception as e:
                logger.exception(f"[async] 摘要补写失败 sid={sid}")
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] [async] 补写异常: {e}")
            finally:
                self._summary_tasks.pop(sid, None)

        self._summary_tasks[sid] = asyncio.create_task(_run())

    async def _do_reset_with_summary(
        self, sid: str, keep_turns: int, reason: str = "超限"
    ) -> List[dict]:
        """执行摘要+重开的核心逻辑（只改磁盘记忆），返回重开后的扁平新历史。

        可被 token 检查（llm_request 钩子）和手动命令共同调用；
        调用方如需同步本轮请求，再自行 _replace_request_messages(req, new_flat)。
        """
        if self.session_mgr is None:
            return []

        now = time.time()
        last_reset = self._last_reset_time.get(sid, 0)

        logger.warning(f"🚨 {sid} {reason}，触发重开，保留最近 {keep_turns} 轮")
        if self.enable_summary_logging:
            logger.info(f"[摘要调试] 摘要模式={self.summarize_mode}, 保留轮数={keep_turns}")

        old_flat = self.session_mgr.fetch_memory(sid)
        old_len = len(old_flat)
        logger.info(f"旧历史消息数: {old_len}")

        # 使用清理后的 chunk 分割
        old_chunks = self._clean_and_chunk(old_flat)
        if not old_chunks:
            logger.warning("历史无法分割成有效 chunk，将直接清空会话")
            new_chunks: List[List[dict]] = []
        else:
            new_chunks = self._extract_recent_chunks(old_chunks, keep_turns)
            logger.info(f"保留的回合数: {len(new_chunks)} (最近 {keep_turns} 轮)")

        # 重开前摘要：优先检查能否复用上次摘要（降级窗口内）
        dropped = self._dropped_flat(old_chunks, keep_turns)
        summary_chunk: Optional[list] = None

        if self.enable_summary_logging:
            logger.info(f"[摘要调试] 待删除历史: {len(dropped)} 条消息")

        # 检查能否复用上次摘要（连续降级时避免重复调 LLM）
        reused_summary = None
        if last_reset and (now - last_reset) < (self.check_interval // 2):
            # 窗口内连续重开，尝试从首条提取已有摘要
            if old_chunks and old_chunks[0]:
                first_msg = old_chunks[0][0]
                if first_msg.get("role") == "user":
                    content = first_msg.get("content", "")
                    if content.startswith("[前情摘要|系统注入]"):
                        # 提取摘要文本（去掉标记行和后面的引导语）
                        lines = content.split("\n", 1)
                        reused_summary = lines[1].strip() if len(lines) > 1 else ""
                        if reused_summary:
                            logger.info(f"♻️ 连续降级，复用上次摘要 ({len(reused_summary)} 字符)")
                            if self.enable_summary_logging:
                                logger.info(f"[摘要调试] 复用的摘要内容:\n{reused_summary}")

        if reused_summary:
            summary_chunk = build_summary_chunk(reused_summary)
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] 使用复用摘要，跳过 LLM 调用")
        elif dropped and self.summarize_mode == "sync":
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] 准备调用 LLM 生成摘要 (sync模式)")
            summary = await self._summarize_dropped(sid, dropped)
            if summary:
                summary_chunk = build_summary_chunk(summary)
                logger.info(f"✅ [sync] 重开前摘要已生成 ({len(summary)} 字符)")
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] 生成的摘要内容:\n{summary}")
            else:
                logger.info("[sync] 摘要不可用，按原逻辑直接重开")
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] LLM 调用失败或返回空，直接重开")
        elif self.summarize_mode == "off":
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] 摘要模式=off，跳过摘要生成")

        if summary_chunk:
            new_chunks = [summary_chunk] + list(new_chunks)

        new_flat = self._flatten_chunks(new_chunks)

        # 先删除会话；随后 get_session_info 重建空会话元数据
        # （write_memory 直接对 chat_memory[sid] 下标赋值，删除后必须先 ensure）
        self.session_mgr.delete_session(sid)
        logger.info(f"✅ 会话 {sid} 已删除")
        self.session_mgr.get_session_info(sid)

        if new_flat:
            chunks_to_write = self._clean_and_chunk(new_flat)  # 重新分块
            if chunks_to_write:
                self.session_mgr.write_memory(sid, chunks_to_write)
                logger.info(
                    f"✅ 新历史已写入: {len(new_flat)} 条消息 -> {len(chunks_to_write)} 个chunk"
                    f"{'（含前情摘要）' if summary_chunk else ''}"
                )
            else:
                self.session_mgr.write_memory(sid, [])
                logger.warning("新历史无法成块，已写入空列表")
        else:
            self.session_mgr.write_memory(sid, [])
            logger.info("新历史为空，已写入空列表")

        # async 模式：只对「未复用摘要」的情况才后台生成
        if dropped and self.summarize_mode == "async" and not reused_summary:
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] 启动后台异步摘要任务")
            self._schedule_async_summary(sid, dropped)
        elif self.summarize_mode == "async" and reused_summary:
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] async模式但已复用摘要，跳过后台任务")

        self._last_reset_time[sid] = now
        logger.info(f"✅ 会话 {sid} 重开完成 (保留轮数={keep_turns})")
        return new_flat

    # ── 压缩重开命令（对齐 reboot_all：整句匹配 + 白名单 + 不进上下文）──

    def _extract_text(self, event: KiraMessageEvent) -> str:
        return "".join(
            elem.text for elem in event.message.chain if isinstance(elem, Text)
        ).strip()

    def _match_command(self, text: str, commands: List[str]) -> bool:
        if not text or not commands:
            return False
        t = text.strip().lower()
        for c in commands:
            if t == str(c).strip().lower():
                return True
        return False

    def _get_event_user_id(self, event: KiraMessageEvent) -> str:
        try:
            return str(event.message.sender.user_id)
        except Exception:
            return "unknown"

    def _reset_command_user_allowed(self, event: KiraMessageEvent) -> bool:
        """参考 reboot_plugin：未开权限 / 白名单为空 → 放行。"""
        if not self.reset_enable_permission:
            return True
        if not self.reset_allowed_users:
            return True
        return self._get_event_user_id(event) in self.reset_allowed_users

    async def _reply(self, sid: str, text: str):
        await self.ctx.message_processor.send_message_chain(
            session=sid,
            chain=MessageChain([Text(text)]),
        )

    @on.im_message(priority=Priority.HIGH)
    async def on_im_message_reset_command(self, event: KiraMessageEvent, *_):
        """压缩重开命令：整句匹配 → 立即压缩重开 → 直接回复，不进入 LLM 上下文。"""
        if not self.enable_reset_command or self.session_mgr is None:
            return
        if not self.reset_commands:
            return

        text = self._extract_text(event)
        if not text or not self._match_command(text, self.reset_commands):
            return

        sid = event.session.sid
        # 命令消息不进入上下文（与其他自定义命令一致）
        event.discard(force=True)
        event.stop()

        user_id = self._get_event_user_id(event)
        logger.info(f"🔑 压缩重开命令触发 | user={user_id} | sid={sid}")

        if not self._reset_command_user_allowed(event):
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] 压缩重开被拒绝 user={user_id}")
            await self._reply(sid, self.reset_permission_denied_message)
            return

        keep_turns = self.keep_recent_turns
        try:
            # 标记本次手动重开，避免 llm_request 钩子在同一时刻又自动重开
            self._last_reset_time[sid] = time.time()
            new_flat = await self._do_reset_with_summary(
                sid, keep_turns, reason=f"压缩重开命令 [{text}]"
            )
            has_summary = bool(new_flat) and str(
                (new_flat[0] or {}).get("content", "")
            ).startswith("[前情摘要|系统注入]")
            if has_summary:
                summary_part = "，已注入前情摘要"
            elif self.summarize_mode == "off":
                summary_part = ""
            elif self.summarize_mode == "async":
                summary_part = "，摘要将在后台生成后补写"
            else:
                summary_part = "（无更早历史可摘要或摘要生成失败）"
            await self._reply(
                sid,
                self.reset_success_message.format(
                    keep=keep_turns, summary=summary_part
                ),
            )
        except Exception as e:
            logger.exception(f"压缩重开失败 sid={sid}")
            await self._reply(sid, self.reset_error_message.format(error=str(e)))

    @on.llm_request(priority=Priority.HIGH)
    async def maybe_reset_session(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if self.session_mgr is None:
            return

        sid = event.sid

        now = time.time()
        # 刚刚（手动命令/上次自动）重开过：跳过本次 token 检查，避免重复重开
        last_reset = self._last_reset_time.get(sid, 0)
        if last_reset and (now - last_reset) < 5:
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] {sid} 刚刚重开过，跳过本轮 token 检查")
            return

        if now - self.last_check.get(sid, 0) < self.check_interval:
            return
        self.last_check[sid] = now

        total_tokens = 0
        for msg in req.messages:
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            total_tokens += self.count_tokens(content)

        if total_tokens < self.max_tokens:
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] {sid} token={total_tokens} < {self.max_tokens}，未超限，跳过重开")
            return

        # 动态降级逻辑
        last_reset = self._last_reset_time.get(sid, 0)
        if last_reset and (now - last_reset) < (self.check_interval // 2):
            old_turns = self._dynamic_keep_turns.get(sid, self.keep_recent_turns)
            new_turns = max(1, old_turns // 2)
            if new_turns < old_turns:
                logger.warning(f"连续超限，保留轮数从 {old_turns} 降级为 {new_turns}")
                self._dynamic_keep_turns[sid] = new_turns
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] 触发降级：距上次重开 {now - last_reset:.1f}s < {self.check_interval // 2}s (窗口)")
            else:
                self._dynamic_keep_turns[sid] = self.keep_recent_turns
        else:
            self._dynamic_keep_turns[sid] = self.keep_recent_turns

        keep_turns = self._dynamic_keep_turns[sid]
        new_flat = await self._do_reset_with_summary(
            sid, keep_turns, reason=f"token={total_tokens} 超过阈值 {self.max_tokens}"
        )
        self._replace_request_messages(req, new_flat)
