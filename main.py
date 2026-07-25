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

from .compat import handle_ccs_conflict
from .summarizer import (
    SUMMARY_MARKER,
    CumulativeSummaryStore,
    build_summary_chunk,
    dropped_fingerprint,
    extract_summary_text,
    is_summary_chunk,
    merge_summaries,
    self_compress_summary,
    summarize_history,
    DEFAULT_SUMMARIZE_PROMPT,
)


class AutoDeleteSessionPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.last_check = defaultdict(float)

        # 兼容旧版：这些字段原在 cfg 顶层，现在拆分到 section_basic / section_reset。
        basic = cfg.get("section_basic", {})
        reset = cfg.get("section_reset", {})

        self.max_tokens = int(basic.get("max_tokens", cfg.get("max_tokens", 10000)))
        self.chars_per_token = float(basic.get("chars_per_token", cfg.get("chars_per_token", 2.0)))
        self.check_interval = int(basic.get("check_interval_seconds", cfg.get("check_interval_seconds", 60)))
        self.keep_recent_turns = int(basic.get("keep_recent_turns", cfg.get("keep_recent_turns", 3)))

        # 写穿式重开：不 delete_session，直接 write_memory 覆写（保留会话标题等元信息）
        self.write_through = bool(reset.get("write_through", cfg.get("write_through", True)))

        # 触发模式：tokens=估算 token 触发；rounds=对齐框架轮数窗口；either=先到先触发
        self.trigger_mode = str(reset.get("trigger_mode", cfg.get("trigger_mode", "tokens")) or "tokens").lower()
        if self.trigger_mode not in ("tokens", "rounds", "either"):
            self.trigger_mode = "tokens"
        _tr = reset.get("trigger_rounds", cfg.get("trigger_rounds", 0))
        self.trigger_rounds = int(_tr if _tr is not None else 0)
        # 缓存修复：把 time 段从 system prompt 挪到请求队尾（随当前消息），
        # 否则时间每分钟变化会在 system 末尾截断前缀缓存，整段记忆永远不进缓存
        self.move_time_to_tail = bool(reset.get("move_time_to_tail", cfg.get("move_time_to_tail", True)))

        # 重开前摘要（兼容旧版全部在 section_summarize；新版拆分到多个分组）
        # 新分组优先：optimize > preprocess > cumulative > summarize，
        # 这样旧版 section_summarize 里的残留值不会覆盖用户在 WebUI 新分组里设置的值。
        sec: dict = {}
        for sec_name in (
            "section_summary_optimize",
            "section_summary_preprocess",
            "section_summary_cumulative",
            "section_summarize",
        ):
            sub = cfg.get(sec_name, {})
            if isinstance(sub, dict):
                for k, v in sub.items():
                    sec.setdefault(k, v)
        self.summarize_mode = str(sec.get("summarize_mode", "sync") or "sync").lower()
        if self.summarize_mode not in ("off", "sync", "async"):
            self.summarize_mode = "sync"
        self.summarize_model = str(sec.get("summarize_model", "") or "")
        self.summarize_timeout_sec = float(sec.get("summarize_timeout_sec", 60.0) or 60.0)
        # 注意：0 是合法值（表示无上限），不能用 `or 默认值` 读取
        _mic = sec.get("summarize_max_input_chars", 10000)
        self.summarize_max_input_chars = int(_mic if _mic is not None else 10000)
        _moc = sec.get("summarize_max_output_chars", 5000)
        self.summarize_max_output_chars = int(_moc if _moc is not None else 5000)
        self.summarize_prompt_template = str(
            sec.get("summarize_prompt_template", "") or ""
        ) or DEFAULT_SUMMARIZE_PROMPT
        self.enable_summary_logging = bool(sec.get("enable_summary_logging", False))

        # 摘要输入预处理（借鉴 CCS preprocessor：超长 tool 结果/图片描述先压缩再摘要）
        self.preprocess_tool_results = bool(sec.get("preprocess_tool_results", True))
        _tmc = sec.get("tool_result_max_chars", 2000)
        self.tool_result_max_chars = int(_tmc if _tmc is not None else 2000)
        # sync 关键路径默认不做预处理（预处理可能耗时数分钟，只留在预热/async 后台）
        self.preprocess_in_sync = bool(sec.get("preprocess_in_sync", False))
        # 合并/自压缩独立超时（比摘要更短，超时快速退化为拼接，避免阻塞回复）
        self.merge_timeout_sec = float(sec.get("merge_timeout_sec", 120.0) or 120.0)

        # 累积摘要（借鉴 CCS：旧摘要 + 新增量合并，不再层层有损叠加）
        self.cumulative_summary = bool(sec.get("cumulative_summary", True))
        self.merge_prompt_template = str(sec.get("merge_prompt_template", "") or "")
        self.self_compress_prompt_template = str(
            sec.get("self_compress_prompt_template", "") or ""
        )

        # 持续后台压缩：超过阈值后每轮回复后都压缩，重开时直接收割
        self.preheat_summary = bool(sec.get("preheat_summary", True))
        try:
            self.preheat_ratio = float(sec.get("preheat_ratio", 0.7))
        except (TypeError, ValueError):
            self.preheat_ratio = 0.7
        if self.preheat_ratio < 0:
            self.preheat_ratio = 0.0
        if self.preheat_ratio > 1:
            self.preheat_ratio = 1.0
        try:
            self.sync_wait_timeout = float(sec.get("sync_wait_timeout", 0.0))
        except (TypeError, ValueError):
            self.sync_wait_timeout = 0.0
        if self.sync_wait_timeout < 0:
            self.sync_wait_timeout = 0.0

        # CCS 风格持续压缩合并策略
        self.continuous_merge_strategy = str(
            sec.get("continuous_merge_strategy", "append_then_merge") or "append_then_merge"
        ).lower()
        if self.continuous_merge_strategy not in ("immediate", "append_then_merge", "append_only"):
            self.continuous_merge_strategy = "append_then_merge"
        self.background_merge_max_concurrent = max(
            1, int(sec.get("background_merge_max_concurrent", 5) or 5)
        )
        self.background_merge_timeout_sec = float(
            sec.get("background_merge_timeout_sec", 120.0) or 120.0
        )
        self.replace_concat_after_merge = bool(sec.get("replace_concat_after_merge", True))
        self.background_merge_retry_interval_sec = float(
            sec.get("background_merge_retry_interval_sec", 30.0) or 30.0
        )
        self.background_merge_max_retries = max(
            0, int(sec.get("background_merge_max_retries", 3) or 3)
        )
        self.max_concat_summary_chars = max(
            500, int(sec.get("max_concat_summary_chars", 5000) or 5000)
        )
        self.concat_overflow_strategy = str(
            sec.get("concat_overflow_strategy", "self_compress") or "self_compress"
        ).lower()
        if self.concat_overflow_strategy not in ("self_compress", "truncate_fifo", "none"):
            self.concat_overflow_strategy = "self_compress"

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

        # 与 ContextCondensation（CCS）的互斥策略
        compat_sec = cfg.get("section_compat", {})
        if not isinstance(compat_sec, dict):
            compat_sec = {}
        self.ccs_conflict_policy = str(
            compat_sec.get("ccs_conflict_policy", "disable_ccs") or "disable_ccs"
        ).lower()
        if self.ccs_conflict_policy not in ("disable_ccs", "disable_self", "ignore"):
            self.ccs_conflict_policy = "disable_ccs"

        self.session_mgr = None
        self._dynamic_keep_turns: Dict[str, int] = {}
        self._last_reset_time: Dict[str, float] = {}
        self._summary_tasks: Dict[str, asyncio.Task] = {}
        # per-sid 锁：手动命令与自动触发、async 补写互斥
        self._locks: Dict[str, asyncio.Lock] = {}
        # 重开后仍超预算标记：驱动下次重开降级 keep（比旧时间窗降级可靠）
        self._token_still_over: Dict[str, bool] = {}
        # 持续后台压缩：sid -> Task / sid -> {"fp", "final", "base", "compressed_len"}
        self._preheat_tasks: Dict[str, asyncio.Task] = {}
        self._preheat_pending: Dict[str, dict] = {}
        # 限制并发后台压缩任务数，避免多个会话同时抢 LLM 导致彼此超时
        self._preheat_sem = asyncio.Semaphore(self.background_merge_max_concurrent)
        self._summary_store: Optional[CumulativeSummaryStore] = None
        self._conflict_disabled = False
        # 后台持续合并任务：sid -> Task
        self._background_merge_tasks: Dict[str, asyncio.Task] = {}
        # 后台合并成功后等待替换框架记忆的新摘要：sid -> final_summary
        self._pending_replace: Dict[str, str] = {}

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

        # 累计摘要存储（插件数据目录持久化）
        try:
            store_path = self.ctx.get_plugin_data_dir() / "cumulative_summaries.json"
            self._summary_store = CumulativeSummaryStore(store_path)
            self._summary_store.load()
        except Exception as e:
            logger.warning(f"AutoDelete: 累计摘要存储初始化失败，降级为非累积模式: {e}")
            self._summary_store = None

        # 与 CCS 的互斥
        try:
            result = await handle_ccs_conflict(
                getattr(self.ctx, "plugin_mgr", None), self.ccs_conflict_policy, logger
            )
            if result == "self_disabled":
                self._conflict_disabled = True
        except Exception as e:
            logger.warning(f"AutoDelete: CCS 冲突处理失败: {e}")

        logger.info(
            f"AutoDeletePlugin 初始化完成: max_tokens={self.max_tokens}, "
            f"chars_per_token={self.chars_per_token}, check_interval={self.check_interval}s, "
            f"keep_recent_turns={self.keep_recent_turns}, summarize={self.summarize_mode}, "
            f"write_through={self.write_through}, cumulative={self.cumulative_summary}, "
            f"continuous_compress={self.preheat_summary}@{self.preheat_ratio}, "
            f"sync_wait={self.sync_wait_timeout}s, "
            f"trigger={self.trigger_mode}(rounds_limit={self._rounds_limit()}), "
            f"preprocess={self.preprocess_tool_results}@{self.tool_result_max_chars}, "
            f"ccs_policy={self.ccs_conflict_policy}"
            f"{' [与CCS冲突已停用]' if self._conflict_disabled else ''}"
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
        self._token_still_over.clear()
        for tasks in (self._summary_tasks, self._preheat_tasks, self._background_merge_tasks):
            for t in list(tasks.values()):
                if t and not t.done():
                    t.cancel()
            tasks.clear()
        self._preheat_pending.clear()
        self._pending_replace.clear()
        if self._summary_store is not None:
            try:
                self._summary_store.save()
            except Exception:
                pass
        self._locks.clear()
        logger.info("AutoDeletePlugin 已终止")

    def _get_lock(self, sid: str) -> asyncio.Lock:
        if sid not in self._locks:
            self._locks[sid] = asyncio.Lock()
        return self._locks[sid]

    def count_tokens(self, text: Any) -> int:
        if not isinstance(text, str):
            text = str(text)
        return max(1, int(len(text) / self.chars_per_token) + 1)

    def _rounds_limit(self) -> int:
        """rounds 触发的轮数上限：配置优先，0 = 对齐框架 max_memory_length。"""
        if self.trigger_rounds > 0:
            return self.trigger_rounds
        return max(1, int(getattr(self.session_mgr, "max_memory_length", 10) or 10))

    @staticmethod
    def _move_time_to_tail(req: LLMRequest) -> bool:
        """把 time 段从 system prompt 挪到 user_prompt 最前（随当前消息）。

        system prompt 因此跨分钟稳定，前缀缓存得以覆盖整段记忆；
        persist=False 保证时间文本不落进会话记忆。幂等：已移动则跳过。
        """
        try:
            sys_prompts = getattr(req, "system_prompt", None) or []
            time_p = None
            for p in sys_prompts:
                if getattr(p, "name", None) == "time":
                    time_p = p
                    break
            if time_p is None:
                return False
            from core.prompt_manager import Prompt

            sys_prompts.remove(time_p)
            req.user_prompt.insert(
                0,
                Prompt(
                    getattr(time_p, "content", "") or "",
                    name="time",
                    source=getattr(time_p, "source", "system") or "system",
                    persist=False,
                ),
            )
            return True
        except Exception:
            return False

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
        """重开后同步本轮请求：整体替换为重开后的历史。

        ON_LLM_REQUEST 阶段 req.messages 只有记忆（system 与当前 user 在
        assemble_prompt 才追加），无需保留其他角色。
        """
        req.messages = [OpenAIMessage(**msg) for msg in new_history]
        logger.info(f"本次请求 messages 已替换: 共 {len(req.messages)} 条")

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

    def _split_head_summary(self, flat: List[dict]) -> tuple:
        """剥离扁平记忆首条的摘要消息，返回 (摘要正文, 剩余消息)。"""
        if flat and isinstance(flat[0], dict):
            first = flat[0]
            if first.get("role") == "user":
                content = str(first.get("content", "") or "")
                if content.startswith(SUMMARY_MARKER):
                    return extract_summary_text(content), list(flat[1:])
        return "", list(flat)

    @staticmethod
    def _effective_dropped(
        head_text: str, dropped: List[dict], reused_head: bool
    ) -> List[dict]:
        """非复用场景下把旧摘要并入摘要输入（对齐旧版「旧摘要随 dropped 再压缩」语义）。

        reused_head=True（累积模式 base 非空 / 降级窗口复用）时旧摘要已在 base 里，
        无需再喂给 LLM。
        """
        if not dropped or not head_text or reused_head:
            return dropped
        return [
            {"role": "user", "content": f"[更早的旧摘要，供参考] {head_text}"}
        ] + list(dropped)

    async def _summarize_dropped(
        self, sid: str, dropped: List[dict], preprocess: Optional[bool] = None
    ) -> Optional[str]:
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
            preprocess_tools=(
                self.preprocess_tool_results if preprocess is None else preprocess
            ),
            tool_max_chars=self.tool_result_max_chars,
        )

    async def _harvest_continuous_compression(
        self, sid: str, timeout: Optional[float] = None
    ) -> Optional[str]:
        """sync 路径：等待并收割持续后台压缩结果。

        - append_then_merge / append_only：直接取 pending["final"] 作为摘要，
          不再做同步 LLM 合并。
        - immediate：把 parts 同步最终合并到只剩一个（尽量复刻 CCS）。
        返回最新 final 摘要，或 None（未就绪）。
        """
        timeout = timeout if timeout is not None else self.sync_wait_timeout
        task = self._preheat_tasks.get(sid)
        if task is not None:
            if not task.done():
                if timeout <= 0:
                    return (self._preheat_pending.get(sid) or {}).get("final")
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
            self._preheat_tasks.pop(sid, None)

        pend = self._preheat_pending.get(sid)
        if not pend:
            return None

        if self.continuous_merge_strategy == "immediate":
            parts = list(pend.get("parts") or [])
            if not parts and pend.get("final"):
                parts = [pend["final"]]
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                merged = await self._reduce_parts(
                    sid, parts, timeout_sec=self.merge_timeout_sec
                )
                if merged:
                    pend["parts"] = [merged]
                    pend["final"] = merged
                    return merged
                # 同步最终合并失败：回退到拼接版，避免再触发一次 LLM delta 摘要
                final = "\n".join(parts)
                pend["final"] = final
                return final
            elif len(parts) == 1:
                pend["final"] = parts[0]
                return parts[0]
            return None

        return pend.get("final")

    async def _cap_summary(self, sid: str, text: str) -> str:
        """累计摘要超输出上限：先 LLM 自压缩，失败则硬截断。"""
        cap = self.summarize_max_output_chars
        text = (text or "").strip()
        if not text or cap <= 0 or len(text) <= cap:
            return text
        compressed = await self_compress_summary(
            self.ctx,
            sid,
            text,
            model_id=self.summarize_model,
            prompt_template=self.self_compress_prompt_template,
            timeout_sec=self.merge_timeout_sec,
            logger=logger,
            enable_detail_log=self.enable_summary_logging,
        )
        if compressed:
            text = compressed.strip()
        if len(text) > cap:
            text = text[:cap] + "…"
        return text

    async def _merge_final(self, sid: str, base: str, delta: Optional[str]) -> str:
        """base（旧累计）+ delta（新增量）→ 新累计摘要；LLM 合并失败退化为拼接。"""
        base = (base or "").strip()
        delta = (delta or "").strip()
        if not base:
            return await self._cap_summary(sid, delta)
        if not delta:
            return base
        if self.enable_summary_logging:
            logger.info(
                f"[摘要调试] 尝试合并摘要 sid={sid} old={len(base)} 字符 "
                f"new={len(delta)} 字符 timeout={self.merge_timeout_sec}s"
            )
        merged = await merge_summaries(
            self.ctx,
            sid,
            base,
            delta,
            model_id=self.summarize_model,
            prompt_template=self.merge_prompt_template,
            timeout_sec=self.merge_timeout_sec,
            logger=logger,
            enable_detail_log=self.enable_summary_logging,
        )
        if not merged:
            merged = f"{base}\n{delta}"
            if self.enable_summary_logging:
                logger.info("[摘要调试] LLM 合并失败，退化为拼接")
        return await self._cap_summary(sid, merged)

    def _build_fused_chunks(
        self, kept_chunks: List[List[dict]], final_summary: Optional[str]
    ) -> List[List[dict]]:
        """缓存友好布局：摘要融合进第一个保留 chunk，不占窗口槽位、位置恒定。"""
        chunks = [list(c) for c in kept_chunks]
        final_summary = (final_summary or "").strip()
        if not final_summary:
            return chunks
        summary_msg = build_summary_chunk(final_summary)[0]
        if chunks:
            chunks[0] = [summary_msg] + chunks[0]
        else:
            chunks = [[summary_msg]]
        return chunks

    def _write_back(self, sid: str, new_chunks: List[List[dict]]):
        """写回新记忆。write_through（默认）：直接覆写，保留 title/description；
        否则走旧的 delete + 重建路径。"""
        if self.write_through:
            data = getattr(self.session_mgr, "chat_memory", None)
            if not isinstance(data, dict) or sid not in data:
                # write_memory 是下标赋值，session 必须先存在
                self.session_mgr.get_session_info(sid)
            self.session_mgr.write_memory(sid, new_chunks)
            logger.info(f"✅ 新历史已覆写（write-through）: {len(new_chunks)} 个chunk")
            return
        self.session_mgr.delete_session(sid)
        logger.info(f"会话 {sid} 已删除（旧路径）")
        self.session_mgr.get_session_info(sid)
        self.session_mgr.write_memory(sid, new_chunks)
        logger.info(f"✅ 新历史已写入: {len(new_chunks)} 个chunk")

    # ── 后台合并辅助 ─────────────────────────────────────────────

    async def _pair_merge_parts(
        self,
        sid: str,
        parts: List[str],
        timeout_sec: Optional[float] = None,
    ) -> List[str]:
        """对 parts 做一次从左到右的 pair merge；成功则合并，失败则保留原片段。

        返回合并后的新 parts 列表，长度 <= len(parts)。
        """
        if not parts:
            return []
        timeout_sec = timeout_sec if timeout_sec is not None else self.merge_timeout_sec
        merged_parts: List[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                merged = await merge_summaries(
                    self.ctx,
                    sid,
                    parts[i],
                    parts[i + 1],
                    model_id=self.summarize_model,
                    prompt_template=self.merge_prompt_template,
                    timeout_sec=timeout_sec,
                    logger=logger,
                    enable_detail_log=self.enable_summary_logging,
                )
                if merged:
                    merged_parts.append(merged.strip())
                    i += 2
                    if self.enable_summary_logging:
                        logger.info(
                            f"[摘要调试] [pair merge] {sid} 合并两段 "
                            f"({len(parts[i - 2])} + {len(parts[i - 1])} -> {len(merged_parts[-1])} 字符)"
                        )
                    continue
            merged_parts.append(parts[i])
            i += 1
        return merged_parts

    async def _reduce_parts(
        self,
        sid: str,
        parts: List[str],
        timeout_sec: Optional[float] = None,
    ) -> Optional[str]:
        """反复 pair merge 直到只剩一个；若无法继续则返回 None。"""
        parts = [p.strip() for p in parts if p.strip()]
        timeout_sec = timeout_sec if timeout_sec is not None else self.background_merge_timeout_sec
        while len(parts) > 1:
            new_parts = await self._pair_merge_parts(sid, parts, timeout_sec=timeout_sec)
            if len(new_parts) >= len(parts):
                # 没有发生任何合并，无法继续
                return None
            parts = new_parts
        return parts[0] if parts else None

    async def _run_background_merge(self, sid: str):
        """后台持续合并：把 pending 中的未合并/落单 parts 继续合并或自压缩。

        受 _preheat_sem 限制并发；失败按间隔重试，最多重试 background_merge_max_retries 次。
        """
        retries = 0
        while True:
            try:
                async with self._preheat_sem:
                    pend = self._preheat_pending.get(sid)
                    if not pend:
                        return
                    captured_fp = pend.get("fp")
                    parts = list(pend.get("parts") or [])
                    if not parts and pend.get("final"):
                        parts = [pend["final"]]
                    parts = [p.strip() for p in parts if p.strip()]
                    if not parts:
                        return

                    new_final: Optional[str] = None
                    if len(parts) > 1:
                        new_final = await self._reduce_parts(
                            sid, parts, timeout_sec=self.background_merge_timeout_sec
                        )
                    if not new_final:
                        concat = "\n".join(parts)
                        if len(parts) > 1 or len(concat) > self.max_concat_summary_chars:
                            new_final = await self_compress_summary(
                                self.ctx,
                                sid,
                                concat,
                                model_id=self.summarize_model,
                                prompt_template=self.self_compress_prompt_template,
                                timeout_sec=self.background_merge_timeout_sec,
                                logger=logger,
                                enable_detail_log=self.enable_summary_logging,
                            )

                    if new_final:
                        new_final = new_final.strip()
                        # 写回 pending 前检查 generation 是否一致，避免覆盖新数据
                        current_pend = self._preheat_pending.get(sid)
                        if current_pend and current_pend.get("fp") == captured_fp:
                            current_pend["parts"] = [new_final]
                            current_pend["final"] = new_final
                        if self.replace_concat_after_merge:
                            self._pending_replace[sid] = new_final
                        if self.enable_summary_logging:
                            logger.info(
                                f"[摘要调试] [后台合并] {sid} 成功，新摘要 {len(new_final)} 字符"
                            )
                        return

                    # 未能取得进展，进入重试
                    if self.enable_summary_logging:
                        logger.info(
                            f"[摘要调试] [后台合并] {sid} 本轮无进展，准备重试"
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] [后台合并] {sid} 异常: {e}")

            if retries >= self.background_merge_max_retries:
                if self.enable_summary_logging:
                    logger.info(f"[摘要调试] [后台合并] {sid} 已达最大重试次数，放弃")
                return
            retries += 1
            await asyncio.sleep(self.background_merge_retry_interval_sec)

    def _schedule_background_merge(self, sid: str):
        """安排后台合并任务；已有任务在运行则跳过。"""
        task = self._background_merge_tasks.get(sid)
        if task is not None and not task.done():
            return
        self._background_merge_tasks[sid] = asyncio.create_task(
            self._run_background_merge(sid)
        )

    async def _self_compress_run(self, sid: str):
        """append_only 超长时的后台自压缩兜底。"""
        try:
            async with self._preheat_sem:
                pend = self._preheat_pending.get(sid)
                if not pend:
                    return
                captured_fp = pend.get("fp")
                final = pend.get("final") or ""
                if len(final) <= self.max_concat_summary_chars:
                    return
                compressed = await self_compress_summary(
                    self.ctx,
                    sid,
                    final,
                    model_id=self.summarize_model,
                    prompt_template=self.self_compress_prompt_template,
                    timeout_sec=self.background_merge_timeout_sec,
                    logger=logger,
                    enable_detail_log=self.enable_summary_logging,
                )
                if compressed:
                    compressed = compressed.strip()
                    current_pend = self._preheat_pending.get(sid)
                    if current_pend and current_pend.get("fp") == captured_fp:
                        current_pend["parts"] = [compressed]
                        current_pend["final"] = compressed
                    if self.replace_concat_after_merge:
                        self._pending_replace[sid] = compressed
                    if self.enable_summary_logging:
                        logger.info(
                            f"[摘要调试] [自压缩] {sid} 成功，"
                            f"{len(final)} -> {len(compressed)} 字符"
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] [自压缩] {sid} 异常: {e}")

    def _schedule_self_compress(self, sid: str):
        """安排后台自压缩任务；已有任务在运行则跳过。"""
        task = self._background_merge_tasks.get(sid)
        if task is not None and not task.done():
            return
        self._background_merge_tasks[sid] = asyncio.create_task(
            self._self_compress_run(sid)
        )

    # ── 持续后台压缩 ───────────────────────────────────────────────

    def _should_compress_continuously(
        self, sid: str, total_tokens: int, rounds: int
    ) -> bool:
        """判断是否超过阈值，需要启动持续后台压缩。"""
        if not self.preheat_summary or self.summarize_mode == "off":
            return False
        if self.preheat_ratio <= 0:
            return False
        rounds_limit = self._rounds_limit()
        if self.trigger_mode in ("tokens", "either"):
            if total_tokens >= self.max_tokens * self.preheat_ratio:
                return True
        if self.trigger_mode in ("rounds", "either"):
            if rounds >= rounds_limit * self.preheat_ratio:
                return True
        return False

    def _schedule_continuous_compression(self, sid: str):
        """启动或维持后台持续压缩任务。"""
        if not self.preheat_summary or self.summarize_mode == "off":
            return
        # 重开持锁期间不启动新任务（结果必定过期）
        lock = self._locks.get(sid)
        if lock is not None and lock.locked():
            return
        task = self._preheat_tasks.get(sid)
        if task is not None and not task.done():
            return
        self._preheat_tasks[sid] = asyncio.create_task(self._continuous_compress_run(sid))

    async def _continuous_compress_run(self, sid: str):
        """持续后台压缩：增量追加新 dropped 消息，避免每轮都重算整段历史。

        只读记忆 + LLM 调用，绝不写会话记忆；失败静默，下一轮继续。
        思路借鉴 CCS：维护已压缩前缀长度，新增消息只需生成 delta 再合并。
        """
        try:
            async with self._preheat_sem:
                if self.session_mgr is None:
                    return
                flat = self.session_mgr.fetch_memory(sid) or []
                head_text, rest = self._split_head_summary(flat)
                chunks = self._clean_and_chunk(rest)
                keep = self._dynamic_keep_turns.get(sid, self.keep_recent_turns)
                dropped = self._dropped_flat(chunks, keep)
                if not dropped:
                    self._preheat_pending.pop(sid, None)
                    return

                if self.cumulative_summary and self._summary_store is not None:
                    base = self._summary_store.sync_with_head(sid, head_text)
                else:
                    base = ""
                eff_dropped = self._effective_dropped(head_text, dropped, reused_head=bool(base))
                fp = dropped_fingerprint(eff_dropped)
                pending = self._preheat_pending.get(sid)

                # 已是最新：无需再压缩
                if pending and pending.get("fp") == fp and (pending.get("base") or "") == (base or ""):
                    return

                # 判断能否增量：base 一致且已压缩长度 < 当前 dropped 长度
                is_incremental = False
                delta_input = eff_dropped
                if pending and (pending.get("base") or "") == (base or ""):
                    prev_len = int(pending.get("compressed_len", 0))
                    if 0 < prev_len < len(dropped):
                        is_incremental = True
                        delta_input = dropped[prev_len:]

                delta = await self._summarize_dropped(
                    sid, delta_input, preprocess=self.preprocess_tool_results
                )
                if not delta:
                    return

                # 维护 parts 片段列表：增量则继承，全量从 base 开始
                if is_incremental:
                    parts = list(pending.get("parts") or [])
                    if not parts and pending.get("final"):
                        parts = [pending["final"]]
                else:
                    parts = [base] if base else []
                parts = [p.strip() for p in parts if p.strip()]
                parts.append(delta.strip())

                # append_only 不调用 merge；其他策略做一轮 pair merge
                if self.continuous_merge_strategy != "append_only":
                    # 后台 pair merge 用后台超时，避免被同步 merge 超时截断
                    parts = await self._pair_merge_parts(
                        sid, parts, timeout_sec=self.background_merge_timeout_sec
                    )

                final = "\n".join(parts)

                # append_only 超长兜底策略
                if self.continuous_merge_strategy == "append_only":
                    if len(final) > self.max_concat_summary_chars:
                        if self.concat_overflow_strategy == "self_compress":
                            self._schedule_self_compress(sid)
                        elif self.concat_overflow_strategy == "truncate_fifo":
                            while (
                                len("\n".join(parts)) > self.max_concat_summary_chars
                                and len(parts) > 1
                            ):
                                parts.pop(0)
                            final = "\n".join(parts)
                        # "none"：不处理

                self._preheat_pending[sid] = {
                    "fp": fp,
                    "final": final,
                    "base": base,
                    "compressed_len": len(dropped),
                    "parts": parts,
                }

                # 还有落单片段时，后台继续合并（append_then_merge 在触发点也会再安排）
                if self.continuous_merge_strategy != "append_only" and len(parts) > 1:
                    self._schedule_background_merge(sid)

                if self.enable_summary_logging:
                    logger.info(
                        f"[摘要调试] [持续压缩] {sid} 摘要已更新 ({len(final)} 字符, "
                        f"{'增量' if is_incremental else '全量'}, "
                        f"strategy={self.continuous_merge_strategy}, parts={len(parts)})"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] [持续压缩] {sid} 异常（静默）: {e}")
        finally:
            self._preheat_tasks.pop(sid, None)

    # ── async 模式补写 ─────────────────────────────────────────

    def _schedule_async_summary(self, sid: str, dropped: List[dict]):
        """async / append fallback 模式：重开先行（带旧累计摘要），后台生成增量并合并补写。"""
        old_task = self._summary_tasks.get(sid)
        if old_task and not old_task.done():
            old_task.cancel()
        # 避免与旧后台合并任务竞争写记忆：以 async 全量摘要为准
        old_bg = self._background_merge_tasks.get(sid)
        if old_bg and not old_bg.done():
            old_bg.cancel()
        self._preheat_pending.pop(sid, None)

        async def _run():
            try:
                delta = await self._summarize_dropped(sid, dropped)
                if not delta or self.session_mgr is None:
                    return
                async with self._get_lock(sid):
                    chunks = self.session_mgr.read_memory(sid) or []
                    head_text = ""
                    if chunks and is_summary_chunk(chunks[0]):
                        head_text = extract_summary_text(
                            str(chunks[0][0].get("content", "") or "")
                        )
                    if self.cumulative_summary and self._summary_store is not None:
                        base = self._summary_store.sync_with_head(sid, head_text)
                    else:
                        base = head_text
                    final = await self._merge_final(sid, base, delta)
                    if not final:
                        return
                    summary_msg = build_summary_chunk(final)[0]
                    if chunks and is_summary_chunk(chunks[0]):
                        # 融合布局：替换 chunk 首条摘要，保留后续消息
                        chunks[0] = [summary_msg] + list(chunks[0][1:])
                    elif chunks:
                        chunks[0] = [summary_msg] + list(chunks[0])
                    else:
                        chunks = [[summary_msg]]
                    self.session_mgr.write_memory(sid, chunks)
                    if self.cumulative_summary and self._summary_store is not None:
                        self._summary_store.set(sid, final)
                        self._summary_store.save()
                    logger.info(f"✅ [async] 累计摘要已补写入会话 {sid} 首部")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.exception(f"[async] 摘要补写失败 sid={sid}")
            finally:
                self._summary_tasks.pop(sid, None)

        self._summary_tasks[sid] = asyncio.create_task(_run())

    # ── 重开核心 ───────────────────────────────────────────────

    async def _do_reset_with_summary(
        self, sid: str, keep_turns: int, reason: str = "超限"
    ) -> Optional[List[dict]]:
        """摘要+重开（只改磁盘记忆），返回重开后的扁平新历史；失败返回 None。

        per-sid 锁保证手动命令/自动触发/async 补写互斥；
        failsafe：任何异常不动磁盘、不动 req。
        """
        if self.session_mgr is None or self._conflict_disabled:
            return None
        async with self._get_lock(sid):
            try:
                return await self._do_reset_locked(sid, keep_turns, reason)
            except Exception:
                logger.exception(f"重开失败（已保护现场，记忆与请求均未改动） sid={sid}")
                return None

    async def _do_reset_locked(
        self, sid: str, keep_turns: int, reason: str
    ) -> List[dict]:
        now = time.time()
        last_reset = self._last_reset_time.get(sid, 0)

        # 钳制：rounds 触发下 keep 必须 < 窗口，否则重开后下一次 append
        # 会被框架抢先截掉融合头部（摘要丢失 + 前缀缓存逐轮失效）
        if self.trigger_mode in ("rounds", "either"):
            cap = max(1, self._rounds_limit() - 1)
            if keep_turns > cap:
                logger.warning(
                    f"保留轮数 {keep_turns} >= rounds 窗口 {self._rounds_limit()}，"
                    f"钳制为 {cap} 防止框架截断头部摘要"
                )
                keep_turns = cap

        logger.warning(f"🚨 {sid} {reason}，触发重开，保留最近 {keep_turns} 轮")
        if self.enable_summary_logging:
            logger.info(f"[摘要调试] 摘要模式={self.summarize_mode}, 保留轮数={keep_turns}")

        old_flat = self.session_mgr.fetch_memory(sid) or []
        logger.info(f"旧历史消息数: {len(old_flat)}")

        # 1) 剥离头部摘要（摘要不再占保留轮数）
        head_text, rest = self._split_head_summary(old_flat)
        old_chunks = self._clean_and_chunk(rest)
        new_chunks = self._extract_recent_chunks(old_chunks, keep_turns)
        dropped = self._dropped_flat(old_chunks, keep_turns)

        # 2) 累计 base：store 与记忆头部对账（防旧摘要污染新会话）
        if self.cumulative_summary and self._summary_store is not None:
            base = self._summary_store.sync_with_head(sid, head_text)
        else:
            # 非累积模式：维持旧语义，仅降级窗口内复用头部摘要
            base = ""
            if last_reset and (now - last_reset) < (self.check_interval // 2):
                base = head_text
                if base:
                    logger.info(f"♻️ 连续降级，复用上次摘要 ({len(base)} 字符)")

        if self.enable_summary_logging:
            logger.info(
                f"[摘要调试] 待删除历史: {len(dropped)} 条, base={len(base)} 字符"
            )

        # 3) 持续后台压缩 → 收割；否则 sync 现场生成 delta；async 延后
        eff_dropped = self._effective_dropped(head_text, dropped, reused_head=bool(base))
        final: Optional[str] = None
        delta: Optional[str] = None
        pending_used = False
        fp = dropped_fingerprint(eff_dropped) if eff_dropped else ""
        if eff_dropped and self.summarize_mode != "off":
            # sync 模式：等待后台压缩完成（最多 sync_wait_timeout 秒）
            if self.summarize_mode == "sync":
                harvested = await self._harvest_continuous_compression(sid)
                if harvested:
                    pend = self._preheat_pending.get(sid)
                    if pend and pend.get("fp") == fp and (pend.get("base") or "") == (base or ""):
                        final = harvested
                        pending_used = bool(final)
                        if pending_used:
                            logger.info("♻️ 收割持续压缩摘要，本次重开零 LLM 调用")
                            if self.enable_summary_logging:
                                logger.info(f"[摘要调试] 持续压缩摘要内容:\n{final}")
                        # append_then_merge 保留 pending，让后台继续合并落单片段
                        if self.continuous_merge_strategy == "append_then_merge":
                            parts = pend.get("parts") or []
                            if len(parts) > 1:
                                self._schedule_background_merge(sid)
                        else:
                            self._preheat_pending.pop(sid, None)
                if final is None:
                    # append_then_merge / append_only：触发点不调用同步 LLM，
                    # 直接靠后台/async 补写完整摘要，避免请求路径阻塞。
                    if self.continuous_merge_strategy in ("append_then_merge", "append_only"):
                        if self.enable_summary_logging:
                            logger.info(
                                f"[摘要调试] {sid} 后台未就绪，"
                                f"strategy={self.continuous_merge_strategy}，"
                                f"触发点不写同步摘要，依赖后台/async 补写"
                            )
                        final = base or None
                    else:
                        # immediate：维持 CCS 式语义，同步生成 delta 并合并
                        delta = await self._summarize_dropped(
                            sid, eff_dropped, preprocess=self.preprocess_in_sync
                        )
            else:
                # async 模式：直接收割已完成的结果，没完成走兜底
                pend = self._preheat_pending.get(sid)
                if pend and pend.get("fp") == fp and (pend.get("base") or "") == (base or ""):
                    final = pend.get("final")
                    pending_used = bool(final)
                    # append_then_merge 保留 pending 供后台继续合并
                    if self.continuous_merge_strategy == "append_then_merge":
                        parts = pend.get("parts") or []
                        if len(parts) > 1:
                            self._schedule_background_merge(sid)
                    else:
                        self._preheat_pending.pop(sid, None)

        # 4) 合并为累计摘要
        if final is None:
            if delta:
                final = await self._merge_final(sid, base, delta)
            else:
                final = base or None

        # 5) 融合布局写回
        fused = self._build_fused_chunks(new_chunks, final)
        self._write_back(sid, fused)

        # 6) 累计摘要落盘
        if self.cumulative_summary and self._summary_store is not None:
            if final:
                self._summary_store.set(sid, final)
            else:
                self._summary_store.pop(sid)
            self._summary_store.save()

        # 7) 后台补 delta 合并（重开已带 base 先行）
        # sync 模式下 append 策略 fallback 时也走 async 补写，避免请求路径阻塞
        should_schedule_async = (
            eff_dropped
            and not pending_used
            and (
                self.summarize_mode == "async"
                or (
                    self.summarize_mode == "sync"
                    and self.continuous_merge_strategy in ("append_then_merge", "append_only")
                )
            )
        )
        if should_schedule_async:
            self._schedule_async_summary(sid, eff_dropped)

        # append_then_merge 且确实用了 pending 时，保留 pending 供后台继续合并
        if not (pending_used and self.continuous_merge_strategy == "append_then_merge"):
            self._preheat_pending.pop(sid, None)
        self._last_reset_time[sid] = now
        logger.info(
            f"✅ 会话 {sid} 重开完成 (保留轮数={keep_turns}"
            f"{'，含累计摘要' if final else ''})"
        )
        return self._flatten_chunks(fused)

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
        if self._conflict_disabled or not self.reset_commands:
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
            if new_flat is None:
                await self._reply(
                    sid, self.reset_error_message.format(error="内部错误（详见日志）")
                )
                return
            # 手动重开后同样记录是否仍超预算（驱动后续自动降级）
            est = 0
            for m in new_flat:
                est += self.count_tokens(m.get("content", ""))
            self._token_still_over[sid] = est >= self.max_tokens
            has_summary = bool(new_flat) and str(
                (new_flat[0] or {}).get("content", "")
            ).startswith(SUMMARY_MARKER)
            if has_summary:
                summary_part = "，已注入累计摘要"
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

    @on.step_result(priority=Priority.LOW)
    async def post_reply_continuous_compress(self, event, *_):
        """每次回复后检查是否需要继续后台压缩，并替换后台合并成功的新摘要。"""
        if not self.preheat_summary or self.summarize_mode == "off":
            return
        sid = getattr(event, "sid", None) or getattr(getattr(event, "session", None), "sid", None)
        if not sid or self.session_mgr is None:
            return
        try:
            total_tokens = 0
            flat = self.session_mgr.fetch_memory(sid) or []
            for msg in flat:
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                total_tokens += self.count_tokens(content)
            rounds = self.session_mgr.get_memory_count(sid)
            if self._should_compress_continuously(sid, total_tokens, rounds):
                self._schedule_continuous_compression(sid)

            # 后台合并/自压缩成功后替换框架记忆中的摘要 chunk
            new_summary = self._pending_replace.pop(sid, None)
            if new_summary:
                async with self._get_lock(sid):
                    chunks = self.session_mgr.read_memory(sid) or []
                    if chunks and is_summary_chunk(chunks[0]):
                        summary_msg = build_summary_chunk(new_summary)[0]
                        chunks[0] = [summary_msg] + list(chunks[0][1:])
                        self.session_mgr.write_memory(sid, chunks)
                        if self.cumulative_summary and self._summary_store is not None:
                            self._summary_store.set(sid, new_summary)
                            self._summary_store.save()
                        logger.info(
                            f"✅ [后台合并] 会话 {sid} 摘要已替换为合并版 "
                            f"({len(new_summary)} 字符)"
                        )
        except Exception:
            pass

    @on.llm_request(priority=Priority.HIGH)
    async def maybe_reset_session(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if self.session_mgr is None or self._conflict_disabled:
            return

        # 缓存修复：time 段挪到队尾（幂等），让 system+记忆成为稳定前缀
        if self.move_time_to_tail:
            self._move_time_to_tail(req)

        sid = event.sid

        now = time.time()
        # 刚刚（手动命令/上次自动）重开过：跳过本次 token 检查，避免重复重开
        last_reset = self._last_reset_time.get(sid, 0)
        if last_reset and (now - last_reset) < 5:
            if self.enable_summary_logging:
                logger.info(f"[摘要调试] {sid} 刚刚重开过，跳过本轮 token 检查")
            return

        # 锁内检查 + 重开：消除「检查在锁外」导致的并发双重重开
        async with self._get_lock(sid):
            try:
                if now - self.last_check.get(sid, 0) < self.check_interval:
                    return
                self.last_check[sid] = now

                total_tokens = 0
                for msg in req.messages:
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    total_tokens += self.count_tokens(content)

                rounds = self.session_mgr.get_memory_count(sid)
                rounds_limit = self._rounds_limit()
                keep_now = self._dynamic_keep_turns.get(sid, self.keep_recent_turns)

                # 双触发：tokens（估算硬预算）与 rounds（对齐框架轮数窗口）
                over = False
                reason = ""
                tokens_over = (
                    self.trigger_mode in ("tokens", "either")
                    and total_tokens >= self.max_tokens
                )
                if tokens_over:
                    if keep_now <= 1 and rounds <= 1:
                        # 唯一真正无可删的场景：keep=1 且只剩 1 轮，
                        # 再重开只会空转重写摘要+毁缓存
                        if self.enable_summary_logging:
                            logger.info(f"[摘要调试] {sid} token 超限但 keep=1 且仅 1 轮，无可压缩，跳过")
                        self._token_still_over[sid] = False
                    else:
                        over = True
                        reason = f"token={total_tokens} 超过阈值 {self.max_tokens}"
                else:
                    self._token_still_over[sid] = False
                if not over and self.trigger_mode in ("rounds", "either"):
                    if rounds >= rounds_limit:
                        over = True
                        reason = f"轮数={rounds} 达到上限 {rounds_limit}"

                if not over:
                    # 接近阈值：启动持续后台压缩，重开时直接收割
                    if self._should_compress_continuously(sid, total_tokens, rounds):
                        self._schedule_continuous_compression(sid)
                    if self.enable_summary_logging:
                        logger.info(f"[摘要调试] {sid} token={total_tokens} rounds={rounds} 未超限（mode={self.trigger_mode}），跳过重开")
                    return

                # 动态降级：上次重开后仍超预算 → keep 减半（信号可靠，
                # 不受旧「30s 时间窗」限制）；预算内则回到配置默认
                if self._token_still_over.get(sid):
                    old_turns = self._dynamic_keep_turns.get(sid, self.keep_recent_turns)
                    new_turns = max(1, old_turns // 2)
                    if new_turns < old_turns:
                        logger.warning(f"重开后仍超预算，保留轮数从 {old_turns} 降级为 {new_turns}")
                    self._dynamic_keep_turns[sid] = new_turns
                else:
                    self._dynamic_keep_turns[sid] = self.keep_recent_turns

                keep_turns = self._dynamic_keep_turns[sid]
                new_flat = await self._do_reset_locked(sid, keep_turns, reason=reason)
                if new_flat is not None:
                    self._replace_request_messages(req, new_flat)
                    # 记录重开后是否仍超预算（驱动下次降级；只增 token 的场景
                    # 最多再降 5→2→1 两三次即收敛，不会无限空转）
                    est = 0
                    for m in new_flat:
                        est += self.count_tokens(m.get("content", ""))
                    self._token_still_over[sid] = est >= self.max_tokens
                    if self._token_still_over[sid]:
                        logger.warning(
                            f"重开后估算 token={est} 仍超 {self.max_tokens}，下次触发将降级保留轮数"
                        )
            except Exception:
                logger.exception(f"重开检查失败（已保护现场） sid={sid}")
                return
