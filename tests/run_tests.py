#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADS 集成测试：在真实 KiraAI 框架代码上加载插件并回归关键场景。

覆盖范围（对应 2.1.4 修复）：
  1. 静态规范：manifest / schema / 默认文案与 schema 一致
  2. 锚点工具：message_fingerprint / covered_anchor / _locate_anchor / _delta_after_anchor
  3. 常规重置（tokens 触发、immediate 策略）回归
  4. 真实流程收割（run → append → 重开）：零同步 LLM 调用、直接采用已就绪摘要
  5. 窗口滑动时增量压缩（输入量显著下降）
  6. 原生截断后累计摘要保活（store 保留 + 重开合并）
  7. keep 无条件钳制（tokens 模式）
  8. 摘要头截断内存桥（仅注入请求、不写记忆）
  9. 空会话 /resum 明确反馈
 10. 会话清空 / 删除事件精确清理（真实 EventBus 分发）与 terminate 反订阅
 11. 命令 /resum 常规回归（含钳制后保留轮数反馈）
 12. 工具/推理字段消息重置与装配回归
 13. summarize_mode=off 回归（不产生摘要、不调用 LLM）
 14. 摘要预处理并发化（有界并发 ≤3 / 保序 / 失败回落 / 零待处理快路径）

用法：
    python tests/run_tests.py --framework /path/to/KiraAI
    KIRA_FRAMEWORK_PATH=/path/to/KiraAI python tests/run_tests.py

找不到框架源码时安全跳过（退出码 0）。
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, info: str = "") -> None:
    CHECKS.append((name, bool(cond), str(info)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


def find_framework(arg: str | None) -> Path | None:
    candidates: list[Path] = []
    if arg:
        candidates.append(Path(arg))
    env = os.environ.get("KIRA_FRAMEWORK_PATH")
    if env:
        candidates.append(Path(env))
    for rel in ("../KiraAI", "../../KiraAI", "./KiraAI", "../kira_upstream"):
        candidates.append((PLUGIN_DIR / rel))
    for c in candidates:
        try:
            c = c.resolve()
        except OSError:
            continue
        if (c / "core" / "plugin" / "__init__.py").exists():
            return c
    return None


def load_plugin_module():
    """以框架同款包结构加载插件 main.py（不经过 PluginManager）。"""
    base_pkg = types.ModuleType("plugins")
    base_pkg.__path__ = [str(PLUGIN_DIR.parent)]
    sys.modules.setdefault("plugins", base_pkg)
    pkg = types.ModuleType("plugins.ads_tests")
    pkg.__path__ = [str(PLUGIN_DIR)]
    sys.modules["plugins.ads_tests"] = pkg
    spec = importlib.util.spec_from_file_location("plugins.ads_tests.main", PLUGIN_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugins.ads_tests.main"] = module
    spec.loader.exec_module(module)
    plugin_cls = None
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, module.BasePlugin) and obj is not module.BasePlugin:
            plugin_cls = obj
    if plugin_cls is None:
        raise RuntimeError("AutoDeleteSessionPlugin class not found in main.py")
    return module, plugin_cls


def main() -> int:
    parser = argparse.ArgumentParser(description="ADS 集成测试")
    parser.add_argument("--framework", help="KiraAI 框架源码目录（含 core/）")
    args = parser.parse_args()

    fw = find_framework(args.framework)
    if fw is None:
        print("SKIP: 未找到 KiraAI 框架源码。")
        print("      请用 --framework /path/to/KiraAI 或环境变量 KIRA_FRAMEWORK_PATH 指定。")
        return 0
    print(f"使用框架: {fw}")
    sys.path.insert(0, str(fw))

    data_dir = Path(tempfile.mkdtemp(prefix="ads_tests_", dir="/tmp"))
    from core.utils.path_utils import init_paths
    init_paths(data_dir=str(data_dir))

    # ---- 静态检查（无需框架运行环境）----
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    check("manifest.version 为 semver", bool(re.fullmatch(r"\d+\.\d+\.\d+", str(manifest.get("version", "")))), str(manifest.get("version")))
    check("manifest.core_version 已声明", bool(manifest.get("core_version")))
    check("manifest.tags 非空", bool(manifest.get("tags")))
    check("manifest.icon 文件存在", (PLUGIN_DIR / str(manifest.get("icon", ""))).is_file())
    schema = json.loads((PLUGIN_DIR / "schema.json").read_text(encoding="utf-8"))
    check("schema.section_command 含 reset_empty_message", "reset_empty_message" in schema["section_command"]["fields"])
    has_deprecated_enum = '"type": "enum"' in (PLUGIN_DIR / "schema.json").read_text(encoding="utf-8")
    check("schema 无已废弃的 enum 类型", not has_deprecated_enum)

    # ---- 框架导入 & 真实环境 ----
    from core.config import KiraConfig
    from core.chat.session_manager import SessionManager
    from core.event_bus import EventBus, SystemEvent  # S10 用
    from core.statistics import Statistics

    module, Plugin = load_plugin_module()

    # 默认文案与 schema 一致（防止两侧漂移）
    sum_mod = sys.modules["plugins.ads_tests.summarizer"]
    check(
        "DEFAULT_SUMMARIZE_PROMPT 与 schema 一致",
        sum_mod.DEFAULT_SUMMARIZE_PROMPT == schema["section_summarize"]["fields"]["summarize_prompt_template"]["default"],
    )
    check(
        "DEFAULT_MERGE_PROMPT 与 schema 一致",
        sum_mod.DEFAULT_MERGE_PROMPT == schema["section_summary_cumulative"]["fields"]["merge_prompt_template"]["default"],
    )
    check(
        "DEFAULT_SELF_COMPRESS_PROMPT 与 schema 一致",
        sum_mod.DEFAULT_SELF_COMPRESS_PROMPT == schema["section_summary_cumulative"]["fields"]["self_compress_prompt_template"]["default"],
    )

    cfg = KiraConfig()
    cfg["bot_config"]["bot"]["max_memory_length"] = 50
    cfg["bot_config"]["bot"]["dynamic_prompt_position"] = "system"
    sess = SessionManager(db=None, kira_config=cfg, event_bus=None)

    class FakeLLMClient:
        def __init__(self):
            self.calls: list[int] = []  # 每次调用的输入长度

        async def chat(self, request, **kwargs):
            from core.provider.llm_model import LLMResponse
            prompt = ""
            for m in request.messages:
                c = getattr(m, "content", "")
                if isinstance(c, str):
                    prompt = c
            self.calls.append(len(prompt))
            out = f"SUMMARY#{len(self.calls)}|{len(prompt)}c"
            # 模拟"合并保留旧摘要内容"：合并提示词会把旧摘要回声出来，
            # 用于断言累计链保留（旧实现丢弃 base 时此标记缺失）
            if "【旧摘要】" in prompt:
                try:
                    # 模板正文里也含"【旧摘要】"字样：从最后一次出现处切分取真实旧摘要
                    old_sum = prompt.rsplit("【旧摘要】", 1)[1].split("【新增记录】", 1)[0].strip()
                    out += f"|MERGE({old_sum[:60]})"
                except Exception:
                    pass
            return LLMResponse(out)

    client = FakeLLMClient()

    class FakePluginMgr:
        def has_plugin(self, pid=None): return False
        def is_plugin_enabled(self, pid=None): return False
        async def set_plugin_enabled(self, pid=None, enabled=None): pass

    class FakeMessageProcessor:
        def __init__(self):
            self.sent = []
        async def send_message_chain(self, session=None, chain=None):
            self.sent.append((session, chain))
            return None

    mp = FakeMessageProcessor()
    stats = Statistics()
    bus = EventBus(stats=stats, event_queue=asyncio.Queue(), db=None)

    class FakeCtx:
        pass

    ctx = FakeCtx()
    ctx.session_mgr = sess
    ctx.config = cfg
    ctx.plugin_mgr = FakePluginMgr()
    ctx.message_processor = mp
    ctx.event_bus = bus
    ctx.get_plugin_data_dir = lambda: data_dir / "plugin_data" / "ads"
    ctx.get_llm_client = lambda model_uuid=None, llm_type=None: client
    ctx.get_default_fast_llm_client = lambda: client
    ctx.get_default_llm_client = lambda: client

    def mk_cfg(**over):
        c = {
            "section_basic": {"max_tokens": 400, "chars_per_token": 2.0,
                              "check_interval_seconds": 10, "keep_recent_turns": 3},
            "section_reset": {"write_through": True, "trigger_mode": "tokens",
                              "trigger_rounds": 0, "move_time_to_tail": True},
            "section_summarize": {"summarize_mode": "sync", "summarize_model": "",
                                  "summarize_timeout_sec": 5.0,
                                  "summarize_max_input_chars": 10000,
                                  "summarize_max_output_chars": 5000,
                                  "summarize_prompt_template": "",
                                  "enable_summary_logging": True},
            "section_summary_optimize": {"continuous_merge_strategy": "immediate",
                                         "preheat_summary": False, "preheat_ratio": 0.7,
                                         "background_merge_max_concurrent": 2,
                                         "background_merge_timeout_sec": 5.0,
                                         "merge_timeout_sec": 5.0, "sync_wait_timeout": 0},
            "section_summary_cumulative": {"cumulative_summary": True},
            "section_command": {"enable_reset_command": True, "reset_commands": ["/resum"]},
        }
        for k, v in over.items():
            if isinstance(v, dict) and k in c:
                c[k].update(v)
            else:
                c[k] = v
        return c

    async def seed(sid, n, tag="", size=200):
        for i in range(n):
            sess.update_memory(sid, [
                {"role": "user", "content": f"{tag}Q{i}-" + "x" * size},
                {"role": "assistant", "content": f"{tag}A{i}-" + "y" * size},
            ])

    async def main_run():
        from core.chat import Session, MessageChain, User
        from core.chat.message_utils import (KiraMessageBatchEvent, KiraMessageEvent,
                                             KiraIMMessage)
        from core.chat.message_elements import Text
        from core.adapter.adapter_info import AdapterInfo
        from core.provider.llm_model import LLMRequest as LR
        from core.prompt_manager import Prompt

        # ---- S02 锚点工具 ----
        fp1 = module.message_fingerprint({"role": "user", "content": "hello"})
        fp2 = module.message_fingerprint({"role": "user", "content": "hello"})
        fp3 = module.message_fingerprint({"role": "assistant", "content": "hello"})
        check("S02 fingerprint 稳定且随角色区分", fp1 == fp2 and fp1 != fp3)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        anchor = module.covered_anchor(msgs)
        check("S02 covered_anchor 取最后两条", len(anchor) == 2 and anchor[1] == module.message_fingerprint(msgs[-1]))

        plugin0 = Plugin(ctx, mk_cfg())
        pos = plugin0._locate_anchor(msgs, [module.message_fingerprint(msgs[2]), module.message_fingerprint(msgs[3])])
        check("S02 _locate_anchor 双指纹命中", pos == 3, f"pos={pos}")
        pos2 = plugin0._locate_anchor(msgs, [module.message_fingerprint({"role": "user", "content": "nope"})])
        check("S02 _locate_anchor 未命中返回 -1", pos2 == -1)
        dropped = msgs[:3]
        rest = msgs[:]
        pend = {"anchor": [module.message_fingerprint(msgs[1]), module.message_fingerprint(msgs[2])]}
        delta = plugin0._delta_after_anchor(pend, rest, dropped)
        check("S02 _delta_after_anchor 取锚点后 dropped 后缀", [m["content"] for m in delta] == [], delta)
        delta2 = plugin0._delta_after_anchor(pend, rest, rest[:4])
        check("S02 _delta_after_anchor 含后继消息", [m["content"] for m in delta2] == ["m3"], delta2)
        delta3 = plugin0._delta_after_anchor({"anchor": ["user:deadbeef00"]}, rest, dropped)
        check("S02 _delta_after_anchor 锚点丢失则全量", len(delta3) == len(dropped))

        # ---- S03 常规重置回归 + S04 真实流程收割 ----
        plugin = Plugin(ctx, mk_cfg(**{
            "section_summary_optimize": {"continuous_merge_strategy": "append_then_merge",
                                         "preheat_summary": True, "preheat_ratio": 0.7},
        }))
        await plugin.initialize()
        check("S03 initialize 成功（SessionManager 就绪）", plugin.session_mgr is sess)
        sid3 = "test:dm:s03"
        await seed(sid3, 20, "s")
        plugin._schedule_continuous_compression(sid3)
        t = plugin._preheat_tasks.get(sid3)
        if t:
            await asyncio.wait_for(asyncio.shield(t), 15)
        pend = plugin._preheat_pending.get(sid3)
        check("S03 预压缩产生 pending", bool(pend and pend.get("final")))
        check("S03 pending 带 anchor 字段", bool(pend and pend.get("anchor")))
        pend_final = pend.get("final") if pend else ""
        await seed(sid3, 1, "turn")  # 回合结束追加（状态前移一轮 —— 旧实现收割必失败）
        calls_before = len(client.calls)
        flat = sess.fetch_memory(sid3)
        req = LR(messages=flat[:])
        ev = KiraMessageBatchEvent(message_types=["dm"], timestamp=0,
                                   session=Session(adapter_name="test", session_type="dm", session_id="s03"))
        plugin.last_check.clear(); plugin._last_reset_time.clear(); plugin._token_still_over.clear()
        await plugin.maybe_reset_session(ev, req)
        in_reset = len(client.calls) - calls_before
        head = str(sess.fetch_memory(sid3)[0].get("content", ""))
        check("S04 重开零同步 LLM 调用", in_reset == 0, f"calls={in_reset}")
        check("S04 头部摘要来自收割结果", bool(pend_final) and pend_final[:40] in head)
        await asyncio.sleep(1.2)
        extra = client.calls[calls_before:]
        check("S04 补写仅为小增量", 0 < len(extra) <= 3 and all(size < 1500 for size in extra),
              f"sizes={extra}")
        await plugin.terminate()

        # ---- S05 窗口滑动增量 ----
        cfg["bot_config"]["bot"]["max_memory_length"] = 6
        plugin5 = Plugin(ctx, mk_cfg(**{
            "section_basic": {"keep_recent_turns": 2, "max_tokens": 999999},
            "section_summary_optimize": {"continuous_merge_strategy": "append_then_merge",
                                         "preheat_summary": True, "preheat_ratio": 0.0},
        }))
        await plugin5.initialize()
        sid5 = "test:dm:s05"
        await seed(sid5, 6, "w", size=300)
        c0 = len(client.calls)
        plugin5._schedule_continuous_compression(sid5)
        t = plugin5._preheat_tasks.get(sid5)
        if t:
            await asyncio.wait_for(asyncio.shield(t), 15)
        run1 = client.calls[c0:]
        sess.update_memory(sid5, [{"role": "user", "content": "U6-" + "x" * 300},
                                  {"role": "assistant", "content": "A6-" + "y" * 300}])
        c1 = len(client.calls)
        plugin5._schedule_continuous_compression(sid5)
        t = plugin5._preheat_tasks.get(sid5)
        if t:
            await asyncio.wait_for(asyncio.shield(t), 15)
        run2 = client.calls[c1:]
        check("S05 滑窗后增量输入显著变小", bool(run2) and max(run2) < 1500,
              f"run1={run1} run2={run2}")
        cfg["bot_config"]["bot"]["max_memory_length"] = 50
        await plugin5.terminate()

        # ---- S06/S07/S08：截断保活 / 钳制 / 内存桥 ----
        plugin6 = Plugin(ctx, mk_cfg(**{
            "section_summary_optimize": {"continuous_merge_strategy": "immediate",
                                         "preheat_summary": False},
        }))
        await plugin6.initialize()
        sid6 = "test:dm:s06"
        await seed(sid6, 5, "t")
        await plugin6._do_reset_with_summary(sid6, 3, reason="r1")
        store_file = data_dir / "plugin_data" / "ads" / "cumulative_summaries.json"
        s1 = json.loads(store_file.read_text(encoding="utf-8")).get(sid6, {}).get("summary", "")
        check("S06 首次重开写入累计摘要", bool(s1))
        cfg["bot_config"]["bot"]["max_memory_length"] = 4
        for _ in range(4):
            sess.update_memory(sid6, [{"role": "user", "content": "N"},
                                      {"role": "assistant", "content": "n"}])
        head_missing = not str(sess.fetch_memory(sid6)[0].get("content", "")).startswith("[前情摘要")
        check("S06 原生截断确实吃掉头部摘要", head_missing)
        # S08：截断后的请求收到内存桥（隔离重置路径：抬高阈值 + 清理节流状态）
        req6 = LR(messages=sess.fetch_memory(sid6)[:])
        ev6 = KiraMessageBatchEvent(message_types=["dm"], timestamp=0,
                                    session=Session(adapter_name="test", session_type="dm", session_id="s06"))
        plugin6.max_tokens = 10 ** 12
        plugin6.last_check.clear()
        plugin6._last_reset_time.clear()
        await plugin6.maybe_reset_session(ev6, req6)
        first6 = getattr(req6.messages[0], "content", "") if req6.messages else ""
        mem_head6 = str(sess.fetch_memory(sid6)[0].get("content", ""))
        check("S08 内存桥将摘要注入请求", str(first6).startswith("[前情摘要"))
        check("S08 内存桥不写记忆", not mem_head6.startswith("[前情摘要"))
        # S06b：再次重开时累计信息保留
        await plugin6._do_reset_with_summary(sid6, 3, reason="r2")
        s2 = json.loads(store_file.read_text(encoding="utf-8")).get(sid6, {}).get("summary", "")
        check("S06 截断后累计链保留（旧摘要内容仍在新摘要中）", bool(s2) and (s1[:24] in s2 or "SUMMARY#1" in s2), f"s2={s2[:60]}")
        cfg["bot_config"]["bot"]["max_memory_length"] = 50
        await plugin6.terminate()

        cfg["bot_config"]["bot"]["max_memory_length"] = 5
        plugin7 = Plugin(ctx, mk_cfg(**{
            "section_basic": {"keep_recent_turns": 5},
            "section_summary_optimize": {"continuous_merge_strategy": "immediate", "preheat_summary": False},
        }))
        await plugin7.initialize()
        sid7 = "test:dm:s07"
        await seed(sid7, 8, "c")
        await plugin7._do_reset_with_summary(sid7, 5, reason="clamp")
        n7 = sess.get_memory_count(sid7)
        head7 = str(sess.fetch_memory(sid7)[0].get("content", ""))
        check("S07 tokens 模式下 keep 被钳制到窗口-1", n7 <= 4, f"chunks={n7}")
        check("S07 钳制后头部摘要保留", head7.startswith("[前情摘要"))
        cfg["bot_config"]["bot"]["max_memory_length"] = 50
        await plugin7.terminate()

        # ---- S09 空会话 /resum ----
        plugin9 = Plugin(ctx, mk_cfg())
        await plugin9.initialize()
        sid9 = "test:dm:s09"
        msg9 = KiraIMMessage(message_id="m9", self_id="bot", chain=MessageChain([Text("/resum")]),
                             timestamp=0, sender=User(user_id="s09", nickname="t"))
        ev9 = KiraMessageEvent(message_types=["dm"], timestamp=0, message=msg9,
                               adapter=AdapterInfo(enabled=True, adapter_id="a", name="test", platform="t"))
        mp.sent.clear()
        await plugin9.on_im_message_reset_command(ev9)
        reply9 = str(mp.sent[-1][1].message_list[0].text) if mp.sent else ""
        check("S09 空会话回复为空历史提示", "没有可压缩的历史" in reply9, reply9[:50])
        direct = await plugin9._do_reset_with_summary(sid9, 3, reason="empty")
        check("S09 空会话直接重置返回 None", direct is None)
        await plugin9.terminate()

        # ---- S10 清空/删除事件 ----
        plugin10 = Plugin(ctx, mk_cfg())
        await plugin10.initialize()
        sid10 = "test:dm:s10"
        check("S10 已订阅 session_memory_written", any(
            getattr(h, "__func__", None) is plugin10._on_memory_written_event.__func__
            for h in bus.subscribers.get("session_memory_written", [])))
        plugin10._summary_store.set(sid10, "OLD-SUMMARY")
        plugin10._summary_store.save()
        plugin10._preheat_pending[sid10] = {"fp": "x", "final": "y"}
        # 非空写入（普通记忆更新）不应清理
        await bus.publish(SystemEvent(event_type="session_memory_written", source="test",
                                      payload={"session": sid10, "old_memory": [], "new_memory": [[{"role": "user", "content": "a"}]]}))
        await bus._process_event(bus.event_queue.get_nowait())
        check("S10 非空写入不清理累计摘要", plugin10._summary_store.get(sid10) == "OLD-SUMMARY")
        # 空写入（用户清空会话）→ 清理
        await bus.publish(SystemEvent(event_type="session_memory_written", source="test",
                                      payload={"session": sid10, "old_memory": [[{"role": "user", "content": "a"}]], "new_memory": []}))
        await bus._process_event(bus.event_queue.get_nowait())
        check("S10 清空事件后累计摘要被丢弃", plugin10._summary_store.get(sid10) == "")
        check("S10 清空事件后暂存被丢弃", sid10 not in plugin10._preheat_pending)
        plugin10._summary_store.set(sid10, "OLD-SUMMARY2")
        await bus.publish(SystemEvent(event_type="session_deleted", source="test",
                                      payload={"session": sid10, "old_memory": []}))
        await bus._process_event(bus.event_queue.get_nowait())
        check("S10 删除事件后累计摘要被丢弃", plugin10._summary_store.get(sid10) == "")
        await plugin10.terminate()
        left = len(bus.subscribers.get("session_memory_written", [])) + len(bus.subscribers.get("session_deleted", []))
        check("S10 terminate 后事件反订阅干净", left == 0, f"left={left}")

        # ---- S11 命令 /resum 常规回归（含钳制反馈）----
        cfg["bot_config"]["bot"]["max_memory_length"] = 5
        plugin11 = Plugin(ctx, mk_cfg(**{"section_basic": {"keep_recent_turns": 8}}))
        await plugin11.initialize()
        sid11 = "test:dm:11"
        await seed(sid11, 8, "q")
        msg11 = KiraIMMessage(message_id="m11", self_id="bot", chain=MessageChain([Text("/resum")]),
                              timestamp=0, sender=User(user_id="11", nickname="t"))
        ev11 = KiraMessageEvent(message_types=["dm"], timestamp=0, message=msg11,
                                adapter=AdapterInfo(enabled=True, adapter_id="a", name="test", platform="t"))
        mp.sent.clear()
        await plugin11.on_im_message_reset_command(ev11)
        reply11 = str(mp.sent[-1][1].message_list[0].text) if mp.sent else ""
        check("S11 命令重开成功", "压缩重开" in reply11 and "失败" not in reply11, reply11[:60])
        check("S11 反馈保留轮数为钳制后的 4", "保留最近 4 轮" in reply11, reply11[:60])
        check("S11 成功文案与 schema 默认一致",
              plugin11.reset_success_message == schema["section_command"]["fields"]["reset_success_message"]["default"])
        check("S11 记忆已重写", sess.get_memory_count(sid11) <= 4, f"chunks={sess.get_memory_count(sid11)}")
        cfg["bot_config"]["bot"]["max_memory_length"] = 50
        await plugin11.terminate()

        # ---- S12 工具/推理字段消息重置与装配 ----
        plugin12 = Plugin(ctx, mk_cfg(**{
            "section_summary_optimize": {"continuous_merge_strategy": "immediate", "preheat_summary": False},
        }))
        await plugin12.initialize()
        sid12 = "test:dm:s12"
        for i in range(6):
            sess.update_memory(sid12, [
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": f"t{i}", "type": "function",
                                 "function": {"name": "search", "arguments": "{}"}}],
                 "reasoning_content": f"think{i}"},
                {"role": "tool", "tool_call_id": f"t{i}", "name": "search", "content": "tool result " * 30},
                {"role": "assistant", "content": f"a{i}"},
            ])
        req12 = LR(messages=sess.fetch_memory(sid12)[:])
        req12.system_prompt = [Prompt("SYS", name="role"), Prompt("T", name="time")]
        req12.user_prompt = [Prompt("hello", name="message")]
        ev12 = KiraMessageBatchEvent(message_types=["dm"], timestamp=0,
                                     session=Session(adapter_name="test", session_type="dm", session_id="s12"))
        plugin12.last_check.clear(); plugin12._token_still_over.clear()
        await plugin12.maybe_reset_session(ev12, req12)
        ok_convert = True
        try:
            for m in req12.messages:
                m.to_dict()
        except Exception as exc:  # noqa: BLE001
            ok_convert = False
        req12.assemble_prompt(dynamic_position="system")
        check("S12 工具/推理消息转换无异常", ok_convert)
        check("S12 头部摘要 + 装配正常", str(req12.messages[1].content).startswith("[前情摘要"))
        check("S12 尾消息为当前 user", req12.messages[-1].role == "user")
        await plugin12.terminate()

        # ---- S13 summarize off 回归 ----
        plugin13 = Plugin(ctx, mk_cfg(**{
            "section_summarize": {"summarize_mode": "off"},
            "section_summary_optimize": {"continuous_merge_strategy": "immediate", "preheat_summary": False},
        }))
        await plugin13.initialize()
        sid13 = "test:dm:s13"
        await seed(sid13, 8, "o")
        calls0 = len(client.calls)
        req13 = LR(messages=sess.fetch_memory(sid13)[:])
        ev13 = KiraMessageBatchEvent(message_types=["dm"], timestamp=0,
                                     session=Session(adapter_name="test", session_type="dm", session_id="s13"))
        plugin13.last_check.clear(); plugin13._token_still_over.clear()
        await plugin13.maybe_reset_session(ev13, req13)
        check("S13 off 模式重置不调用 LLM", len(client.calls) - calls0 == 0)
        flat13 = sess.fetch_memory(sid13)
        check("S13 off 模式不产生摘要", not str(flat13[0].get("content", "")).startswith("[前情摘要"))
        await plugin13.terminate()

        # ---- S14 摘要预处理并发化（有界并发 / 保序 / 失败回落）----
        prep = importlib.import_module("plugins.ads_tests.preprocessor")

        class _PreprocessProbe:
            def __init__(self):
                self.inflight = 0
                self.max_inflight = 0
                self.calls = 0

            async def chat(self, request, **kwargs):
                self.inflight += 1
                self.calls += 1
                if self.inflight > self.max_inflight:
                    self.max_inflight = self.inflight
                try:
                    await asyncio.sleep(0.08)
                    prompt = ""
                    for m in request.messages:
                        c = getattr(m, "content", "")
                        if isinstance(c, str):
                            prompt = c
                    if "FAILME" in prompt:
                        raise RuntimeError("probe fail")
                    from core.provider.llm_model import LLMResponse
                    return LLMResponse("PSUM")
                finally:
                    self.inflight -= 1

        probe = _PreprocessProbe()
        long_text = "alpha beta gamma " * 120
        msgs14 = [
            {"role": "tool", "content": "[T1] " + long_text},
            {"role": "user", "content": "普通用户消息"},
            {"role": "tool", "content": "[T2] " + long_text},
            {"role": "tool", "content": "[T3] FAILME " + long_text},
            {"role": "user", "content": "另一个普通用户消息"},
            {"role": "tool", "content": "[T4] " + long_text},
            {"role": "tool", "content": "[T5] " + long_text},
        ]
        orig_first = msgs14[0]["content"]
        out14 = await prep.preprocess_messages_for_summary(msgs14, 200, probe, module.logger)
        check("S14 预处理有界并发（2..3 且确有并行）", 2 <= probe.max_inflight <= 3, f"max={probe.max_inflight}")
        check("S14 调用次数=待处理条数（含失败项）", probe.calls == 5, f"calls={probe.calls}")
        check("S14 未处理项原样保留且顺序不变",
              out14[1] is msgs14[1] and out14[4] is msgs14[4] and len(out14) == len(msgs14))
        check("S14 失败项回落原文", out14[3] is msgs14[3])
        check("S14 处理后项为副本且带压缩后缀",
              out14[0] is not msgs14[0] and str(out14[0].get("content", "")).endswith("（已压缩）")
              and str(out14[6].get("content", "")).endswith("（已压缩）"))
        check("S14 原消息未被改动", msgs14[0]["content"] == orig_first)

        probe2 = _PreprocessProbe()
        out14b = await prep.preprocess_messages_for_summary(
            [{"role": "user", "content": "短消息"}, {"role": "tool", "content": "short"}],
            200, probe2, module.logger,
        )
        check("S14 零待处理快路径（不调用 LLM、原样返回）",
              probe2.calls == 0 and out14b[0]["content"] == "短消息" and out14b[1]["content"] == "short")

    asyncio.run(main_run())

    failed = [c for c in CHECKS if not c[1]]
    print("=" * 70)
    print(f"TOTAL: {len(CHECKS)} checks, {len(failed)} failed")
    for name, _, info in failed:
        print(f"  FAIL: {name} {info}")
    shutil.rmtree(data_dir, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
