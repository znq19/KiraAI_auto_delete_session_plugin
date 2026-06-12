import time
import math
from collections import defaultdict
from typing import Any, List, Dict

from core.plugin import BasePlugin, logger, on, Priority
from core.chat.message_utils import KiraMessageBatchEvent
from core.provider.llm_model import LLMRequest
from core.agent.message import OpenAIMessage


class AutoDeleteSessionPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.last_check = defaultdict(float)
        self.max_tokens = int(cfg.get("max_tokens", 5000))
        self.chars_per_token = float(cfg.get("chars_per_token", 2.0))
        self.check_interval = int(cfg.get("check_interval_seconds", 60))
        self.keep_recent_turns = int(cfg.get("keep_recent_turns", 3))

        self.session_mgr = None
        self._dynamic_keep_turns: Dict[str, int] = {}
        self._last_reset_time: Dict[str, float] = {}

    async def initialize(self):
        if hasattr(self.ctx, 'session_mgr'):
            self.session_mgr = self.ctx.session_mgr
        else:
            self.session_mgr = self._find_session_manager()

        if self.session_mgr is None:
            logger.error("AutoDelete: 无法找到 SessionManager，插件无法工作")
            return

        required = ['fetch_memory', 'write_memory', 'delete_session', 'get_session_info']
        missing = [m for m in required if not hasattr(self.session_mgr, m)]
        if missing:
            logger.error(f"AutoDelete: SessionManager 缺少方法: {missing}")
            self.session_mgr = None
            return

        logger.info(
            f"AutoDeletePlugin 初始化完成: max_tokens={self.max_tokens}, "
            f"chars_per_token={self.chars_per_token}, check_interval={self.check_interval}s, "
            f"keep_recent_turns={self.keep_recent_turns}"
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

    @on.llm_request(priority=Priority.HIGH)
    async def maybe_reset_session(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if self.session_mgr is None:
            return

        sid = event.sid
        now = time.time()
        if now - self.last_check.get(sid, 0) < self.check_interval:
            return
        self.last_check[sid] = now

        total_tokens = 0
        for msg in req.messages:
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            total_tokens += self.count_tokens(content)

        if total_tokens < self.max_tokens:
            logger.debug(f"{sid} token={total_tokens} < {self.max_tokens}，跳过")
            return

        # 动态降级逻辑
        last_reset = self._last_reset_time.get(sid, 0)
        if last_reset and (now - last_reset) < (self.check_interval // 2):
            old_turns = self._dynamic_keep_turns.get(sid, self.keep_recent_turns)
            new_turns = max(1, old_turns // 2)
            if new_turns < old_turns:
                logger.warning(f"连续超限，保留轮数从 {old_turns} 降级为 {new_turns}")
                self._dynamic_keep_turns[sid] = new_turns
            else:
                self._dynamic_keep_turns[sid] = self.keep_recent_turns
        else:
            self._dynamic_keep_turns[sid] = self.keep_recent_turns

        keep_turns = self._dynamic_keep_turns[sid]
        logger.warning(f"🚨 {sid} token={total_tokens} 超过阈值 {self.max_tokens}，触发重开，保留最近 {keep_turns} 轮")

        old_flat = self.session_mgr.fetch_memory(sid)
        old_len = len(old_flat)
        logger.info(f"旧历史消息数: {old_len}")

        # 使用清理后的 chunk 分割
        old_chunks = self._clean_and_chunk(old_flat)
        if not old_chunks:
            logger.warning("历史无法分割成有效 chunk，将直接清空会话")
            new_flat = []
        else:
            new_chunks = self._extract_recent_chunks(old_chunks, keep_turns)
            new_flat = self._flatten_chunks(new_chunks)
            logger.info(f"保留的回合数: {len(new_chunks)} (最近 {keep_turns} 轮)，消息数: {len(new_flat)}")

        # 先删除会话
        self.session_mgr.delete_session(sid)
        logger.info(f"✅ 会话 {sid} 已删除")

        self.session_mgr.get_session_info(sid)

        if new_flat:
            chunks_to_write = self._clean_and_chunk(new_flat)  # 重新分块
            if chunks_to_write:
                self.session_mgr.write_memory(sid, chunks_to_write)
                logger.info(f"✅ 新历史已写入: {len(new_flat)} 条消息 -> {len(chunks_to_write)} 个chunk")
                verify = self.session_mgr.fetch_memory(sid)
                logger.info(f"✅ 验证: 存储中现有消息数 = {len(verify)}")
            else:
                self.session_mgr.write_memory(sid, [])
                logger.warning("新历史无法成块，已写入空列表")
        else:
            self.session_mgr.write_memory(sid, [])
            logger.info("新历史为空，已写入空列表")

        self._replace_request_messages(req, new_flat)

        self._last_reset_time[sid] = now
        logger.info(f"✅ 会话 {sid} 重开完成 (动态保留轮数={keep_turns})")
