"""Versioned, label-free behavior measurements and durable profile records.

Independent implementation of elementary sequence statistics. No fitting,
labels, paths, task names, tool output or model identity enter extraction.
"""
from __future__ import annotations

import gzip
import math
from collections import Counter
from dataclasses import asdict

from .traces import ATOM_GROUP
from .evidence import ModeScore, TraceEvidence

PROFILE_VERSION = "behavior-profile-v3"
ACTION_LABELS = tuple(sorted(set(ATOM_GROUP.values())))
ACTION_ALPHABET_SIZE = len(ACTION_LABELS)
TENSOR_PHASES = 4
TENSOR_CHANNELS = ("command_share", "within_family_vocabulary_diversity", "adjacent_exact_repeat_rate")
# key, Chinese label, rationale, origin; availability is checked per record.
FEATURE_SPECS = (
    ("log_command_count", "命令数量对数", "控制可见操作规模，并检验长度捷径", "project-v1"),
    ("command_ttr", "唯一命令比例", "测量工具词汇复用程度", "AIT-C-adapted/project-v1"),
    ("command_entropy_bits", "命令分布熵", "测量工具使用是否集中，补充只计种类的TTR", "AIT-H-adapted/ProcGrep"),
    ("command_top1_share", "最高频命令占比", "直接定位单一工具依赖程度", "ProcGrep-concentration-adapted"),
    ("adjacent_command_repeat", "相邻命令重复率", "测量局部重复，但不把重复等同于失败重试", "project-v1"),
    ("command_bigram_repeat", "命令二元组重复率", "测量重复的两步操作组合", "project-v1"),
    ("command_bigram_entropy_bits", "命令二元组熵", "区分多种组合与固定组合的频率分布", "AIT-E-adapted"),
    ("command_singleton_bigram_rate", "单次命令二元组占比", "测量一次性组合占全部二元组出现次数的比例", "AIT-E-adapted"),
    ("command_trigram_repeat", "命令三元组重复率", "检验两步以上的短程模式是否有增量", "project-extension"),
    ("command_gzip_ratio", "命令序列压缩比", "低成本概括可重复性；显式检查长度和gzip头开销", "AIT-E-adapted"),
    ("action_coverage", "动作类别覆盖率", "在固定8类动作表上测量行为广度", "project-extension"),
    ("action_entropy_bits", "动作分布熵", "保留粗动作分布，降低对具体工具词汇的依赖", "AIT-H-adapted/ProcGrep"),
    ("action_top1_share", "最高频动作占比", "测量单一行为类别的集中程度", "ProcGrep-concentration-adapted"),
    ("adjacent_action_repeat", "相邻动作重复率", "测量行为类别的局部连续性", "project-v1"),
    ("action_bigram_repeat", "动作二元组重复率", "测量跨工具的重复动作组合", "project-v1"),
    ("action_bigram_entropy_bits", "动作二元组熵", "测量行为转移分布的多样性", "AIT-E-adapted"),
    ("action_singleton_bigram_rate", "单次动作二元组占比", "测量粗动作组合的新颖程度", "AIT-E-adapted"),
    ("mean_action_run_share", "平均动作连续段占比", "区分集中连续操作与频繁切换", "project-v1"),
    ("max_action_run_share", "最长动作连续段占比", "识别轨迹是否被一段连续操作主导", "project-v1"),
    ("action_run_cv", "动作连续段长度变异系数", "补充平均值，测量连续段是否均匀", "project-extension"),
    ("max_command_run_share", "最长同命令连续段占比", "区分同工具连续使用与仅同类动作连续使用", "project-extension"),
    ("command_half_jsd", "前后半段命令分布差异", "检验工具使用是否随执行阶段变化", "ProcGrep-JSD-adapted"),
    ("action_half_jsd", "前后半段动作分布差异", "检验较少依赖命令名的阶段变化", "ProcGrep-JSD-adapted"),
    ("short_trace_flag", "短轨迹标记", "记录不足两条命令，避免把缺少转移当成真实零值", "project-v1"),
)
FEATURE_NAMES = tuple(spec[0] for spec in FEATURE_SPECS)
LENGTH_SENSITIVE = {"log_command_count", "command_gzip_ratio", "short_trace_flag"}


def _entropy(values):
    if not values:
        return None
    n = len(values)
    return -sum((count / n) * math.log2(count / n) for count in Counter(values).values())


def _runs(values):
    output = []
    for index, value in enumerate(values):
        if index and value == values[index - 1]:
            output[-1] += 1
        else:
            output.append(1)
    return output


def _ngrams(values, size):
    return [tuple(values[i:i + size]) for i in range(len(values) - size + 1)]


def _repeat(values):
    return (len(values) - len(set(values))) / len(values) if values else None


def _half_jsd(values):
    if len(values) < 4:
        return None
    middle = len(values) // 2
    first, second = Counter(values[:middle]), Counter(values[middle:])
    result = 0.0
    for item in first.keys() | second.keys():
        p, q = first[item] / middle, second[item] / (len(values) - middle)
        mean = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / mean)
        if q:
            result += 0.5 * q * math.log2(q / mean)
    return result


def extract_behavior_features(commands, actions):
    """Return 24 finite scalars or None; empty/misaligned streams stay explicit.

    Actions may be absent or have their own length. Cross-stream alignment is
    not assumed. Unknown action alphabets invalidate coverage only; other
    action measurements remain descriptive statistics of the supplied stream.
    """
    commands, actions = tuple(commands), tuple(actions)
    if any(not isinstance(item, str) or not item for item in (*commands, *actions)):
        raise ValueError("commands and actions must contain non-empty strings")
    n, m = len(commands), len(actions)
    command_pairs, action_pairs = _ngrams(commands, 2), _ngrams(actions, 2)
    command_runs, action_runs = _runs(commands), _runs(actions)
    counts_c, counts_a = Counter(commands), Counter(actions)
    raw = "\n".join(commands).encode("utf-8")
    mean_run = sum(action_runs) / len(action_runs) if action_runs else None
    known_actions = set(ATOM_GROUP.values())
    values = (
        math.log1p(n), len(counts_c) / n if n else None, _entropy(commands),
        max(counts_c.values()) / n if n else None,
        sum(a == b for a, b in command_pairs) / len(command_pairs) if command_pairs else None,
        _repeat(command_pairs), _entropy(command_pairs),
        sum(count == 1 for count in Counter(command_pairs).values()) / len(command_pairs) if command_pairs else None,
        _repeat(_ngrams(commands, 3)),
        len(gzip.compress(raw, compresslevel=9, mtime=0)) / len(raw) if raw else None,
        len(counts_a) / ACTION_ALPHABET_SIZE if m and set(actions) <= known_actions else None,
        _entropy(actions), max(counts_a.values()) / m if m else None,
        sum(a == b for a, b in action_pairs) / len(action_pairs) if action_pairs else None,
        _repeat(action_pairs), _entropy(action_pairs),
        sum(count == 1 for count in Counter(action_pairs).values()) / len(action_pairs) if action_pairs else None,
        mean_run / m if m else None, max(action_runs) / m if m else None,
        math.sqrt(sum((x - mean_run) ** 2 for x in action_runs) / len(action_runs)) / mean_run if m else None,
        max(command_runs) / n if n else None,
        _half_jsd(commands), _half_jsd(actions), float(n < 2),
    )
    return dict(zip(FEATURE_NAMES, values))


def extract_behavior_tensor(commands, actions):
    """Build a label-free phase × action-family × channel profile.

    Commands and action labels must describe the same ordered events. The
    phase axis uses four contiguous equal-count bins; empty cells have zero
    rates, with command_share distinguishing no observations from zero repeats.
    """
    commands, actions = tuple(commands), tuple(actions)
    if len(commands) != len(actions):
        raise ValueError("tensor profiles require command/action event alignment")
    if any(action not in ACTION_LABELS for action in actions):
        raise ValueError("tensor profile contains an unsupported action family")
    action_index = {action: index for index, action in enumerate(ACTION_LABELS)}
    tensor = [[[0.0 for _ in TENSOR_CHANNELS] for _ in ACTION_LABELS]
              for _ in range(TENSOR_PHASES)]
    n = len(commands)
    for phase in range(TENSOR_PHASES):
        start = (n * phase) // TENSOR_PHASES
        end = (n * (phase + 1)) // TENSOR_PHASES
        width = end - start
        if not width:
            continue
        for action in ACTION_LABELS:
            positions = [index for index in range(start, end) if actions[index] == action]
            if not positions:
                continue
            cell = tensor[phase][action_index[action]]
            cell[0] = len(positions) / width
            cell[1] = len({commands[index] for index in positions}) / len(positions)
            same_action_pairs = [index for index in range(start, end - 1)
                                 if actions[index] == action and actions[index + 1] == action]
            cell[2] = (sum(commands[index] == commands[index + 1] for index in same_action_pairs)
                       / len(same_action_pairs)) if same_action_pairs else 0.0
    return tensor


def build_behavior_profile(evidence: TraceEvidence, score: ModeScore | None = None):
    """Persist evidence and a distinct prediction channel, never training truth."""
    if score is not None and (score.trace_id != evidence.trace_id or score.source_format != evidence.source_format):
        raise ValueError("mode score must match trace_id and source_format")
    features = extract_behavior_features(evidence.commands, evidence.actions)
    try:
        tensor_values = extract_behavior_tensor(evidence.commands, evidence.actions)
        tensor_unavailable = None
    except ValueError as exc:
        tensor_values = None
        tensor_unavailable = str(exc)
    return {
        "schema_version": PROFILE_VERSION,
        "trace_id": evidence.trace_id,
        "source_format": evidence.source_format,
        "feature_values": features,
        "tensor_profile": {
            "schema_version": "phase-action-channel-v1",
            "shape": [TENSOR_PHASES, ACTION_ALPHABET_SIZE, len(TENSOR_CHANNELS)],
            "axes": {
                "phase": ["early", "early_middle", "late_middle", "late"],
                "action_family": list(ACTION_LABELS),
                "channel": list(TENSOR_CHANNELS),
            },
            "available": tensor_values is not None,
            "values": tensor_values,
            "unavailable_reason": tensor_unavailable,
        },
        "unavailable_features": {name: "insufficient_sequence_or_unsupported_action_alphabet"
                                 for name, value in features.items() if value is None},
        "action_proportions": {action: evidence.actions.count(action) / len(evidence.actions)
                               for action in sorted(set(evidence.actions))} if evidence.actions else {},
        "evidence_counts": {"commands": len(evidence.commands), "actions": len(evidence.actions)},
        "event_refs": list(evidence.event_refs),
        "quality": dict(evidence.quality),
        "mode_estimate": asdict(score) if score else {
            "prediction": "unknown", "agent_like_score": None, "calibrated": False,
            "available": False, "reason": "no_format_matched_detector_supplied",
        },
        "interpretation": "Behavior profile, not actor identity; raw model scores are not calibrated participation probabilities.",
    }
