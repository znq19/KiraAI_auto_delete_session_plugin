import time
from collections import defaultdict
from typing import Any

from core.plugin import BasePlugin, logger, on, Priority
from core.chat.message_utils import KiraMessageBatchEvent
from core.provider.llm_model import LLMRequest


class AutoDeleteSessionPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.last_check = defaultdict(float)
        self.max_tokens = int(cfg.get("max_tokens", 50000))
        self.chars_per_token = float(cfg.get("chars_per_token", 4.0))
        self.check_interval = int(cfg.get("check_interval_seconds", 60))
        self.mem_mgr = None

    async def initialize(self):
        self.mem_mgr = self._find_memory_manager()
        if self.mem_mgr is None:
            logger.error("无法找到记忆管理器，插件将不会工作！请检查 ctx 对象。")
        else:
            logger.info(f"找到记忆管理器: {type(self.mem_mgr).__name__}")
            if not hasattr(self.mem_mgr, 'delete_session'):
                logger.error(f"记忆管理器 {type(self.mem_mgr).__name__} 没有 delete_session 方法，插件将不会工作！")
                self.mem_mgr = None

        logger.info(
            f"AutoDeleteSessionPlugin initialized: max_tokens={self.max_tokens}, "
            f"chars_per_token={self.chars_per_token}, check_interval={self.check_interval}s"
        )

    def _find_memory_manager(self):
        direct_names = ['memory_manager', 'session_manager', 'mem_manager', 'memory_mgr', 'session_mgr', 'memory', 'session']
        for name in direct_names:
            if hasattr(self.ctx, name):
                candidate = getattr(self.ctx, name)
                if candidate and self._is_memory_manager(candidate):
                    return candidate

        intermediate_objects = [
            ('message_processor', ['memory_manager', 'session_manager']),
            ('processor', ['memory_manager', 'session_manager']),
            ('bot', ['memory_manager', 'session_manager']),
            ('kira', ['memory_manager', 'session_manager']),
        ]
        for obj_name, sub_names in intermediate_objects:
            if hasattr(self.ctx, obj_name):
                obj = getattr(self.ctx, obj_name)
                for sub in sub_names:
                    if hasattr(obj, sub):
                        candidate = getattr(obj, sub)
                        if candidate and self._is_memory_manager(candidate):
                            return candidate

        logger.info(f"ctx 可用属性: {dir(self.ctx)}")
        return None

    def _is_memory_manager(self, obj):
        return hasattr(obj, 'delete_session') and callable(getattr(obj, 'delete_session'))

    async def terminate(self):
        self.last_check.clear()

    def count_tokens(self, text: Any) -> int:
        if not isinstance(text, str):
            text = str(text)
        return max(1, int(len(text) / self.chars_per_token) + 1)

    @on.llm_request(priority=Priority.LOW)
    async def maybe_delete_session(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if self.mem_mgr is None:
            return

        sid = event.sid
        now = time.time()
        last = self.last_check.get(sid, 0)
        if now - last < self.check_interval:
            return
        self.last_check[sid] = now

        messages = req.messages
        if not messages:
            return

        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            total_tokens += self.count_tokens(content)

        if total_tokens < self.max_tokens:
            return

        try:
            # 注意：delete_session 是同步方法，不要加 await
            self.mem_mgr.delete_session(sid)
            logger.info(f"会话 {sid} 因 token 超限 ({total_tokens} > {self.max_tokens}) 已自动删除")
            req.messages = []  # 清空本次请求的历史
        except Exception as e:
            logger.error(f"删除会话 {sid} 失败: {e}")