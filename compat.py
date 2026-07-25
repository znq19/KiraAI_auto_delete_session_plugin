from __future__ import annotations

"""
与 ContextCondensation（CCS）的互斥处理。

CCS 会在 ON_LLM_REQUEST 改写 req.messages 并通过 write_memory 覆写会话记忆，
与 ADS 的重开写回属于同一职责，并行会互相污染（ADS 丢掉的内容会被 CCS
从自己的缓存里摘要「复活」）。默认策略：自动禁用 CCS。
"""

from typing import Optional

CCS_CANDIDATE_IDS = (
    "KiraAI-ContextCondensation",
    "KiraAI-ContextCondensation-main",
    "context_condensation",
    "context-condensation",
    "ContextCondensation",
)


def find_ccs_plugin_id(plugin_mgr) -> Optional[str]:
    """候选 id 精确匹配 + 模糊扫描（目录名可能带 -main 等后缀）。"""
    if not plugin_mgr:
        return None
    try:
        for pid in CCS_CANDIDATE_IDS:
            if plugin_mgr.has_plugin(pid):
                return pid
        for attr in ("plugin_instances", "plugins", "_plugins"):
            d = getattr(plugin_mgr, attr, None)
            if isinstance(d, dict):
                for key in d:
                    norm = str(key).lower().replace("-", "").replace("_", "")
                    if "contextcondensation" in norm:
                        return str(key)
    except Exception:
        pass
    return None


async def handle_ccs_conflict(plugin_mgr, policy: str, logger) -> str:
    """
    按策略处理与 CCS 的冲突。返回执行结果：
    "none"（无冲突/未处理）| "ccs_disabled" | "self_disabled"
    """
    policy = (policy or "disable_ccs").strip().lower()
    pid = find_ccs_plugin_id(plugin_mgr)
    if not pid:
        return "none"
    try:
        if not plugin_mgr.is_plugin_enabled(pid):
            return "none"
    except Exception:
        return "none"

    if policy == "ignore":
        if logger:
            logger.warning(
                "检测到 %s 已启用（ccs_conflict_policy=ignore）：两者都会覆写会话记忆，"
                "并行可能互相污染上下文，建议只保留一个",
                pid,
            )
        return "none"

    if policy == "disable_self":
        if logger:
            logger.warning(
                "检测到 %s 已启用，按策略 ccs_conflict_policy=disable_self："
                "ADS 本次加载后不生效（ hooks 全部跳过）",
                pid,
            )
        return "self_disabled"

    # 默认 disable_ccs
    try:
        await plugin_mgr.set_plugin_enabled(pid, False)
        if logger:
            logger.warning(
                "已自动禁用 %s（ccs_conflict_policy=disable_ccs）："
                "ADS 2.0 已吸收其累积压缩设计，请不要再同时启用",
                pid,
            )
        return "ccs_disabled"
    except Exception as e:
        if logger:
            logger.warning("禁用 %s 失败: %s；两插件并行可能冲突", pid, e)
        return "none"
