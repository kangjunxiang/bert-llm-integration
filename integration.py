"""
LLM + BERT 实体融合
========================================
实现三种主融合策略及对应消融:
    策略A   Static Weight Fusion         (静态权重融合, 无需训练)
    策略A'' Vote-Aware Hard-Rule Static  (硬规则 + 票数门, 无需训练)
    策略A_v2 Consensus V2 Hard-Rule      (per-beam 投票 + conf 分散度, 无需训练)
    策略A_cal Calibrated Weighted Fusion (Isotonic 校准 + per-type α 加权, 无需训练)
    策略B   Gating Network Fusion        (门控网络融合, 端到端学习)
消融实验 (Beam-1 only):
    策略A_b1    / 策略A_cal_b1    / 策略B_b1
控制台输出每种策略在验证集和测试集上的 P/R/F1。
"""

import json
import math
import os
import ast
import time
import random
import re
import warnings
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')
# 全局唯一种子, 保证实验可复现
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
# 固定 Python hash 种子, 避免 dict 顺序抖动影响结果
if not os.environ.get('PYTHONHASHSEED'):
    os.environ['PYTHONHASHSEED'] = '0'

# ====================================================================
# 0. 全局常量
# ====================================================================
ALL_TYPES = ['dis', 'sym', 'dru', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep']
NUM_TYPES = len(ALL_TYPES)
TYPE2IDX = {t: i for i, t in enumerate(ALL_TYPES)}

# 位置容差: LLM/BERT 预测位置 |diff| <= TOL 视为同一实体
POSITION_TOLERANCE = 2


# ====================================================================
# 1. 数据解析
# ====================================================================
def _safe_parse_json(line: str):
    """安全解析 JSON, 失败时回退到 ast.literal_eval (兼容单引号格式)。"""
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        try:
            return ast.literal_eval(line.strip())
        except Exception:
            return None


def _is_mark_format_line(line: str) -> bool:
    """检测单行是否为 mark 格式 (predict 字段含 ' :' 标记)。

    mark 格式特征: 行内有 ' :' 且在方括号两侧, 例如 "[entity] :type"。
    """
    obj = _safe_parse_json(line)
    if obj is None:
        return False
    try:
        models = obj.get('ner_models', []) or obj.get('llm_beams_ner_models', [])
        if models:
            for m in models:
                predict = m.get('predict', '') or ''
                if ' :' in predict and bool(re.search(r'\[[^\[\]]+\]', predict)):
                    return True
        label = obj.get('label', '') or ''
        if isinstance(label, str) and ' :' in label and bool(re.search(r'\[[^\[\]]+\]', label)):
            return True
    except Exception:
        pass
    return False


def _clean_entity_type(etype: str):
    """清理实体类型, 返回有效类型或 None"""
    VALID = set(ALL_TYPES)
    if etype in VALID:
        return etype
    cleaned = etype.lstrip('-.、。,，)）:： ')
    return cleaned if cleaned in VALID else None


def _split_mark_line(line):
    """将 'etype : content' 拆成 (etype, content), 失败返回 None"""
    line = line.strip()
    if not line or ' :' not in line:
        return None
    parts = line.split(' :', 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1]


def _entities_with_pos_in_content(content):
    """从单行 mark content 中抽取实体 (start, end, entity), start/end 是行内偏移 (剔除 [] 后)。

    兼容两种格式:
      A) 原句 + [entity] 标记  → 精确偏移
      B) 裸实体: '稽留热、弛张热' → 按 [、,;；] 切分, 偏移在 content 内的实际位置
    """
    out = []
    bracket = list(re.finditer(r'\[(.*?)\]', content))
    if bracket:
        offset = 0
        for m in bracket:
            entity = m.group(1).strip()
            if not entity:
                continue
            bs = m.start()
            out.append((bs - offset, bs - offset + len(entity) - 1, entity))
            offset += 2
        return out
    for m in re.finditer(r'[^、,;；]+', content):
        seg = m.group(0).strip()
        if not seg:
            continue
        idx = content.find(seg, m.start())
        if idx < 0:
            idx = m.start()
        out.append((idx, idx + len(seg) - 1, seg))
    return out


def parse_marked_text_with_pos(text: str):
    """解析 mark 格式文本, 返回 [(start, end, entity, type, line_idx), ...]

    位置是剔除 [ 和 ] 后, 在该行内容中的真实索引, 闭区间 (start + len(entity) - 1)。
    兼容 [entity] 标记与裸实体两种格式。
    """
    result = []
    for line_idx, line in enumerate(text.strip().split('\n')):
        r = _split_mark_line(line)
        if not r:
            continue
        head, content = r
        etype = _clean_entity_type(head)
        if etype is None:
            continue
        for start, end, entity in _entities_with_pos_in_content(content):
            result.append((start, end, entity, etype, line_idx))
    return result


def _correct_entity_positions(entities_with_pos, original_text, tolerance=2,
                              keep_all_matches=False):
    """校正位置到原始完整文本中的真实位置。

    规则:
      1. 实体不在 original_text 中 → 丢弃 (LLM 幻觉, 不参与评估)
      2. 实体在 original_text 中, 位置已正确 → 保留
      3. 实体在 original_text 中, 位置不匹配:
         a. 仅 1 处匹配 → 直接使用该位置
         b. 多处匹配:
            - keep_all_matches=False (默认): 启用位置容差 `tolerance`,
              优先选 |match.start - 预测 start| <= tolerance 中距离最近的位置;
              若无容差内匹配, 退而选全部匹配中距离最近的位置
            - keep_all_matches=True: 全部采用, 把每一处匹配都作为候选位置加入
    """
    if not original_text:
        return entities_with_pos
    corrected = []
    for start_idx, end_idx, entity, etype, line_no in entities_with_pos:
        if not entity:
            continue
        if entity not in original_text:
            continue
        if 0 <= start_idx <= end_idx < len(original_text) \
                and original_text[start_idx:end_idx + 1] == entity:
            corrected.append((start_idx, end_idx, entity, etype, line_no))
            continue
        try:
            matches = list(re.finditer(re.escape(entity), original_text))
        except re.error:
            continue
        if not matches:
            continue
        if len(matches) == 1:
            m = matches[0]
            corrected.append((m.start(), m.end() - 1, entity, etype, line_no))
            continue
        if keep_all_matches:
            for m in matches:
                corrected.append((m.start(), m.end() - 1, entity, etype, line_no))
            continue
        within = [m for m in matches if abs(m.start() - start_idx) <= tolerance]
        if within:
            best = min(within, key=lambda m: abs(m.start() - start_idx))
        else:
            best = min(matches, key=lambda m: abs(m.start() - start_idx))
        corrected.append((best.start(), best.end() - 1, entity, etype, line_no))
    return corrected


def _parse_mark_format_predict(predict_text, original_text=None, default_conf=0.8):
    """mark 格式 predict → entity_dict: {(start,end,type,entity): {conf,entity,type,start,end}}"""
    ents = parse_marked_text_with_pos(predict_text)
    ents = _correct_entity_positions(ents, original_text)
    out = {}
    for s, e, txt, t, _ in ents:
        key = (s, e, t, txt)
        if key not in out:
            out[key] = {'confidence': default_conf, 'entity': txt, 'type': t,
                        'start_idx': s, 'end_idx': e}
    return out


# LLM 多 beam 聚合参数 (max+mean+vote_reward 模型)
# 公式:  P = α · max(c_i) + (1-α) · mean(c_i) + β · (N-1)
#  - α: 最佳 beam conf 的权重 (默认 0.7)
#  - β: 投票数线性奖励 (默认 0.05)
#  - N: 预测该实体的 beam 数量
LLM_ALPHA = None      # type: float | None
LLM_VOTE_REWARD = None  # type: float | None


def _aggregate_llm_conf(confs, beam_idxs, alpha, beta):
    """max+mean+vote_reward 聚合多 beam 预测同一实体的置信度。

    P = α · max(c_i) + (1-α) · mean(c_i) + β · (N-1), clamp 到 [0, 1]
    """
    if not confs:
        return 0.0
    a = alpha if alpha is not None else 0.7
    b = beta if beta is not None else 0.05
    score = a * max(confs) + (1 - a) * (sum(confs) / len(confs)) + b * (len(confs) - 1)
    return max(0.0, min(1.0, score))


def _parse_mark_format_beam(models, original_text, default_conf=0.8):
    """多 beam 预测的 max+mean+vote_reward 聚合。

    输入: models = [ {predict: str, confidence: float}, ... ]  (按 beam 排名 1→N)
    输出: { (start,end,type,text): {confidence, vote_count, best_beam_idx,
              best_beam_conf, beam_idxs, beam_confs, llm_avg_conf, llm_max_conf, ...} }
    """
    if not models:
        return {}
    alpha = LLM_ALPHA if LLM_ALPHA is not None else 0.7
    beta = LLM_VOTE_REWARD if LLM_VOTE_REWARD is not None else 0.05
    agg = {}
    for beam_idx, m in enumerate(models):
        beam_conf = float(m.get('confidence', default_conf))
        ents = _parse_mark_format_predict(m.get('predict', ''), original_text, default_conf=default_conf)
        for k, v in ents.items():
            if k not in agg:
                agg[k] = {
                    'beam_idxs': [],
                    'beam_confs': [],
                    'best_idx': beam_idx,
                    'best_conf': beam_conf,
                    'entity': v['entity'], 'type': v['type'],
                    'start_idx': v['start_idx'], 'end_idx': v['end_idx'],
                }
            agg[k]['beam_idxs'].append(beam_idx)
            agg[k]['beam_confs'].append(beam_conf)
            if beam_idx < agg[k]['best_idx']:
                agg[k]['best_idx'] = beam_idx
                agg[k]['best_conf'] = beam_conf

    out = {}
    for k, v in agg.items():
        confs = v['beam_confs']
        idxs = v['beam_idxs']
        vote_count = len(confs)
        best_idx = v['best_idx']
        best_conf = v['best_conf']
        avg_conf = sum(confs) / vote_count
        max_conf = max(confs)
        combined = _aggregate_llm_conf(confs, idxs, alpha, beta)
        out[k] = {
            'confidence': combined,
            'entity': v['entity'], 'type': v['type'],
            'start_idx': v['start_idx'], 'end_idx': v['end_idx'],
            'vote_count': float(vote_count),
            'best_beam_idx': float(best_idx),
            'best_beam_conf': best_conf,
            'beam_idxs': list(idxs),
            'beam_confs': list(confs),
            'llm_avg_conf': avg_conf,
            'llm_max_conf': max_conf,
        }
    return out


def _parse_mark_format_label(label_text):
    """mark 格式 label → [{entity,start_idx,end_idx,type}, ...]"""
    return [
        {'entity': txt, 'start_idx': s, 'end_idx': e, 'type': t}
        for s, e, txt, t, _ in parse_marked_text_with_pos(label_text)
    ]


def _normalize_entity_confidences(entity_dict):
    """归一化置信度到 [0, 1]; 全相同时不动, 避免引入随机噪声。"""
    if not entity_dict:
        return entity_dict
    confs = [v['confidence'] for v in entity_dict.values()]
    min_c, max_c = min(confs), max(confs)
    if max_c - min_c < 0.01:
        return entity_dict
    for v in entity_dict.values():
        v['confidence'] = (v['confidence'] - min_c) / (max_c - min_c)
    return entity_dict


def parse_llm_line(line, source='ner_models', use_beam=False, default_conf=0.8):
    """解析 LLM 文件一行 → (text, entity_dict, label_entities, llm_parse_error)

    use_beam=True:  对多 beam 投票, entity_dict 中包含 vote_count / llm_avg_conf / llm_max_conf
    use_beam=False: 仅用第 1 个 beam, c['llm_conf'] = Beam-1 真实 conf, vote_count=1
    """
    obj = _safe_parse_json(line)
    if obj is None:
        return None
    text = obj['text']
    models = obj.get(source, []) or obj.get('ner_models', [])

    # 解析 label 作 ground truth
    if _is_mark_format_line(line):
        label_text = obj.get('label', '')
        label_entities = _parse_mark_format_label(label_text) if isinstance(label_text, str) else \
            (label_text if isinstance(label_text, list) else [])
    else:
        label_field = obj.get('label', '[]')
        if isinstance(label_field, str):
            le = _safe_parse_json(label_field) if label_field else []
            if le is None:
                le = []
        else:
            le = label_field if isinstance(label_field, list) else []
        label_entities = le

    if not models:
        return text, {}, label_entities, obj.get('llm_parse_error', False)

    if _is_mark_format_line(line):
        if use_beam and len(models) >= 2:
            entity_dict = _parse_mark_format_beam(models, text, default_conf=default_conf)
        else:
            # 单 beam 也走 beam 聚合函数, 让 cand 携带真实 conf + vote_count=1, 统一接口
            entity_dict = _parse_mark_format_beam(models[:1], text, default_conf=default_conf)
        entity_dict = _normalize_entity_confidences(entity_dict)
        return text, entity_dict, label_entities, obj.get('llm_parse_error', False)

    # 标准 JSON 格式
    entity_dict = {}
    for model in models:
        mconf = model.get('confidence', 0.0)
        try:
            ents_str = model.get('predict', '[]')
            ents = _safe_parse_json(ents_str) if isinstance(ents_str, str) else ents_str
            if ents is None:
                continue
        except Exception:
            continue
        for e in ents:
            if not isinstance(e, dict) or 'entity' not in e:
                continue
            etype = e.get('type', '')
            if not etype:
                continue
            s = int(e['start_idx'])
            ee = s + len(e['entity'])
            key = (s, ee, etype, e.get('entity', ''))
            if key not in entity_dict or mconf > entity_dict[key]['confidence']:
                entity_dict[key] = {
                    'confidence': mconf, 'entity': e.get('entity', ''),
                    'type': etype, 'start_idx': s, 'end_idx': ee
                }
    entity_dict = _normalize_entity_confidences(entity_dict)

    label_field = obj.get('label', '[]')
    if isinstance(label_field, str):
        le = _safe_parse_json(label_field) if label_field else []
        if le is None:
            le = []
    else:
        le = label_field if isinstance(label_field, list) else []
    return text, entity_dict, le, obj.get('llm_parse_error', False)


def parse_bert_line(line):
    """解析 BERT 文件一行 → (text, [entity dicts])
    BERT 的 start_idx = label 位置 + 1, 这里减 1 对齐 LLM。
    """
    obj = json.loads(line.strip())
    text = obj['full_text']
    entities = []
    for etype, elist in obj['entities'].items():
        for e in elist:
            entities.append({
                'entity': e[0],
                'start_idx': int(e[1]) - 1,
                'end_idx': int(e[2]),
                'type': etype,
                'confidence': float(e[3])
            })
    return text, entities


# ====================================================================
# 2. 候选生成 / 匹配
# ====================================================================
def _entities_match(e1, e2, tol=0):
    """同 type + 同 text + |start 差| <= tol 视为同一实体"""
    if e1.get('entity', '') != e2.get('entity', ''):
        return False
    if e1.get('type', '') != e2.get('type', ''):
        return False
    if abs(int(e1.get('start_idx', 0)) - int(e2.get('start_idx', 0))) > tol:
        return False
    return True


def _entity_key(e):
    """(start_idx, entity_text, type) — 用于候选 key / 评估"""
    return (e['start_idx'], e['entity'], e['type'])


def build_candidates(text, entity_dict, bert_entities, tol=0):
    """合并 LLM 实体 (entity_dict) 与 BERT 实体, 输出候选列表。

    每条候选: {start_idx, end_idx, type, entity, llm_conf, bert_conf,
                llm_present, bert_present, [可选 beam 特征] ...}
    合并原则: 同一 (start, entity, type) 优先合并; 否则 (type, entity 相同
    且 |start 差| <= tol) 也视为同一候选, BERT 侧匹配最近 LLM 候选。
    """
    cand_dict = {}
    llm_list = list(entity_dict.values())

    for ent in llm_list:
        k = (ent['start_idx'], ent['entity'], ent['type'])
        cand = {
            'start_idx': ent['start_idx'], 'end_idx': ent['end_idx'],
            'type': ent['type'], 'entity': ent['entity'],
            'llm_conf': ent['confidence'], 'bert_conf': 0.0,
            'llm_present': 1, 'bert_present': 0,
        }
        if 'vote_count' in ent:
            cand['vote_count'] = ent['vote_count']
            cand['best_beam_idx'] = ent.get('best_beam_idx', 0.0)
            cand['best_beam_conf'] = ent.get('best_beam_conf', ent['confidence'])
            cand['llm_avg_conf'] = ent['llm_avg_conf']
            cand['llm_max_conf'] = ent['llm_max_conf']
            cand['llm_beam_idxs'] = list(ent.get('beam_idxs', []))
            cand['beam_confs'] = list(ent.get('beam_confs', []))
        cand_dict[k] = cand

    for eb in bert_entities:
        bk = (eb['start_idx'], eb['entity'], eb['type'])
        matched_key = None
        if bk in cand_dict and not cand_dict[bk]['bert_present']:
            matched_key = bk
        elif tol > 0:
            best, best_diff = None, tol + 1
            for ent in llm_list:
                if not _entities_match(eb, ent, tol):
                    continue
                d = abs(int(eb['start_idx']) - int(ent['start_idx']))
                if d < best_diff:
                    best, best_diff = ent, d
                    if d == 0:
                        break
            if best is not None:
                matched_key = (best['start_idx'], best['entity'], best['type'])

        if matched_key is not None:
            cand_dict[matched_key]['bert_conf'] = eb['confidence']
            cand_dict[matched_key]['bert_present'] = 1
        else:
            if bk in cand_dict:
                if eb['confidence'] > cand_dict[bk]['bert_conf']:
                    cand_dict[bk]['bert_conf'] = eb['confidence']
            else:
                cand_dict[bk] = {
                    'start_idx': eb['start_idx'], 'end_idx': eb['end_idx'],
                    'type': eb['type'], 'entity': eb['entity'],
                    'llm_conf': 0.0, 'bert_conf': eb['confidence'],
                    'llm_present': 0, 'bert_present': 1,
                }
    return list(cand_dict.values())


def _label_set(label_entities, tol=0):
    """构建 label key 集合 (位置容差扩展)。"""
    if tol <= 0:
        return {(int(e['start_idx']), e.get('entity', ''), e.get('type', ''))
                for e in label_entities if 'start_idx' in e}
    keys = set()
    for e in label_entities:
        for off in range(-tol, tol + 1):
            keys.add((int(e['start_idx']) + off, e['entity'], e['type']))
    return keys


def _entities_match_in_list(target, candidates, tol=0):
    """在 candidates 中找第一个与 target 匹配的实体"""
    for c in candidates:
        if _entities_match(target, c, tol):
            return c
    return None


def _reaggregate_llm_confs(samples, alpha, beta):
    """按新的 (α, β) 重新算每个 candidate 的 llm_conf。

    不重 build_samples, 直接用候选里已存的 beam_confs 重算, 避免重复解析文本。
    优先使用 c['beam_confs']; 缺失时退化为 c['llm_max_conf'] + c['llm_conf'] 双点近似。
    """
    for s in samples:
        for c in s['candidates']:
            confs = c.get('beam_confs') or None
            if not confs:
                max_c = c.get('llm_max_conf')
                if isinstance(max_c, (int, float)) and max_c > 0:
                    confs = [float(max_c), float(c['llm_conf'])]
                else:
                    confs = [float(c['llm_conf'])]
            n = len(confs)
            if n == 0:
                continue
            base = alpha * max(confs) + (1 - alpha) * (sum(confs) / n)
            score = base + beta * (n - 1)
            c['llm_conf'] = max(0.0, min(1.0, score))


# ====================================================================
# 3. 评估函数
# ====================================================================
def evaluate(preds_list, labels_list, tol=0):
    """startIndexMatch 评估: 统计 (start_idx, entity, type) 三元组集合的命中。

    tol: 位置容差 (论文主评估: 2)。
    """
    tp, total_pred, total_true = _precompute_metrics(preds_list, labels_list, tol)
    tp = int(tp.sum()); total_pred = int(total_pred.sum()); total_true = int(total_true.sum())
    p = tp / total_pred if total_pred else 0.0
    r = tp / total_true if total_true else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, total_pred, total_true


def _precompute_metrics(preds_list, labels_list, tol=0):
    """预计算每个样本的 (tp, |p_set|, |l_set|) → 返回 3 个 np.array。

    用途: bootstrap 重采样时直接向量化和, 不再重算 set 交集。
    """
    n = len(preds_list)
    tps = np.empty(n, dtype=np.int64)
    pns = np.empty(n, dtype=np.int64)
    tns = np.empty(n, dtype=np.int64)
    for i, (preds, labels) in enumerate(zip(preds_list, labels_list)):
        if tol <= 0:
            p_set = {(int(e['start_idx']), e.get('entity', ''), e.get('type', ''))
                     for e in preds if 'start_idx' in e}
            l_set = {(int(e['start_idx']), e.get('entity', ''), e.get('type', ''))
                     for e in labels if 'start_idx' in e}
        else:
            p_set, l_set = set(), set()
            for e in preds:
                if 'start_idx' not in e:
                    continue
                for off in range(-tol, tol + 1):
                    p_set.add((int(e['start_idx']) + off, e.get('entity', ''), e.get('type', '')))
            for e in labels:
                if 'start_idx' not in e:
                    continue
                for off in range(-tol, tol + 1):
                    l_set.add((int(e['start_idx']) + off, e.get('entity', ''), e.get('type', '')))
        tps[i] = len(p_set & l_set)
        pns[i] = len(p_set)
        tns[i] = len(l_set)
    return tps, pns, tns


def bootstrap_f1_ci(preds_list, labels_list, n_boot=1000, alpha=0.05,
                    tol=0, seed=GLOBAL_SEED):
    """Paired Bootstrap 算 F1 的 95% 置信区间。

    方法: 有放回重采样文本索引 n_boot 次, 每次算 F1, 取 [α/2, 1-α/2] 分位数。
    返回: (mean, lo, hi, std)
    """
    rng = np.random.RandomState(seed)
    n = len(preds_list)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    tps, pns, tns = _precompute_metrics(preds_list, labels_list, tol)
    f1s = np.empty(n_boot)
    idx_buf = np.empty(n, dtype=np.int64)
    for b in range(n_boot):
        idx_buf[:] = rng.randint(0, n, size=n)
        tp = tps[idx_buf].sum()
        pn = pns[idx_buf].sum()
        tn = tns[idx_buf].sum()
        p = tp / pn if pn else 0.0
        r = tp / tn if tn else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s[b] = f1
    f1s.sort()
    lo = f1s[int(alpha / 2 * n_boot)]
    hi = f1s[int((1 - alpha / 2) * n_boot) - 1]
    return float(f1s.mean()), float(lo), float(hi), float(f1s.std())


def paired_bootstrap_pvalue(preds_a, labels_a, preds_b, labels_b,
                             n_boot=1000, tol=0, seed=GLOBAL_SEED):
    """Paired Bootstrap 检验 H0: F1(B) - F1(A) = 0。

    返回: (delta_mean, delta_ci_lo, delta_ci_hi, p_value_two_sided)
    """
    rng = np.random.RandomState(seed)
    n = len(preds_a)
    if n == 0 or len(preds_b) != n:
        return 0.0, 0.0, 0.0, 1.0
    tps_a, pns_a, tns_a = _precompute_metrics(preds_a, labels_a, tol)
    tps_b, pns_b, tns_b = _precompute_metrics(preds_b, labels_b, tol)
    return paired_bootstrap_pvalue_from_arr(tps_a, pns_a, tns_a, tps_b, pns_b, tns_b,
                                             n_boot=n_boot, seed=seed)


def paired_bootstrap_pvalue_from_arr(tps_a, pns_a, tns_a, tps_b, pns_b, tns_b,
                                      n_boot=1000, seed=GLOBAL_SEED):
    """paired_bootstrap_pvalue 的预计算数组版本, 多策略共享预计算结果。"""
    rng = np.random.RandomState(seed)
    n = len(tps_a)
    if n == 0 or len(tps_b) != n:
        return 0.0, 0.0, 0.0, 1.0
    deltas = np.empty(n_boot)
    idx_buf = np.empty(n, dtype=np.int64)
    for b in range(n_boot):
        idx_buf[:] = rng.randint(0, n, size=n)
        tp_a = tps_a[idx_buf].sum()
        pn_a = pns_a[idx_buf].sum()
        tn_a = tns_a[idx_buf].sum()
        p_a = tp_a / pn_a if pn_a else 0.0
        r_a = tp_a / tn_a if tn_a else 0.0
        fa = 2 * p_a * r_a / (p_a + r_a) if (p_a + r_a) else 0.0
        tp_b = tps_b[idx_buf].sum()
        pn_b = pns_b[idx_buf].sum()
        tn_b = tns_b[idx_buf].sum()
        p_b = tp_b / pn_b if pn_b else 0.0
        r_b = tp_b / tn_b if tn_b else 0.0
        fb = 2 * p_b * r_b / (p_b + r_b) if (p_b + r_b) else 0.0
        deltas[b] = fb - fa
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot) - 1]
    # 双边 p-value: H1: ΔF1 ≠ 0
    n_le = int((deltas <= 0).sum())
    n_gt = n_boot - n_le
    p_val = float(min(n_le, n_gt) * 2 / n_boot)
    p_val = min(p_val, 1.0)
    return float(deltas.mean()), float(lo), float(hi), p_val


def bootstrap_f1_ci_from_arr(tps, pns, tns, n_boot=1000, alpha=0.05,
                              seed=GLOBAL_SEED):
    """bootstrap_f1_ci 的预计算数组版本, 多策略共享预计算结果。"""
    rng = np.random.RandomState(seed)
    n = len(tps)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    f1s = np.empty(n_boot)
    idx_buf = np.empty(n, dtype=np.int64)
    for b in range(n_boot):
        idx_buf[:] = rng.randint(0, n, size=n)
        tp = tps[idx_buf].sum()
        pn = pns[idx_buf].sum()
        tn = tns[idx_buf].sum()
        p = tp / pn if pn else 0.0
        r = tp / tn if tn else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s[b] = f1
    f1s.sort()
    lo = f1s[int(alpha / 2 * n_boot)]
    hi = f1s[int((1 - alpha / 2) * n_boot) - 1]
    return float(f1s.mean()), float(lo), float(hi), float(f1s.std())


def print_metrics(name, p, r, f1, tp=None, tp_pred=None, tp_true=None):
    extra = ""
    if tp is not None:
        extra = f"  (TP={tp}, pred={tp_pred}, true={tp_true})"
    print(f"  {name:<32} P={p:.4f}  R={r:.4f}  F1={f1:.4f}{extra}")


def _type_only_f1(samples, target_type, source='bert', conf_th=0.5):
    """按 type 评估单模型 F1。
    source='bert' → 只看 BERT-only 候选, 若 bert_conf >= conf_th 则预测
    source='llm'  → 只看 LLM-only + consensus 候选, 若 llm_conf >= conf_th 则预测
    """
    preds_list, labels_list = [], []
    for s in samples:
        kept = []
        for c in s['candidates']:
            if c['type'] != target_type:
                continue
            if source == 'bert':
                if c['bert_present'] and c['bert_conf'] >= conf_th:
                    kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                 'type': c['type'], 'entity': c['entity']})
            else:  # llm
                if c['llm_present'] and c['llm_conf'] >= conf_th:
                    kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                 'type': c['type'], 'entity': c['entity']})
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    _, _, f1, _, _, _ = evaluate(preds_list, labels_list, tol=POSITION_TOLERANCE)
    return f1


def eval_llm_beam_k(llm_lines, bert_lines, beam_idx=0, default_conf=0.8,
                    source='ner_models'):
    """模拟 LLM 只用第 beam_idx 个 beam 预测, 算 P/R/F1。

    关键: ground truth 必须用 LLM 文件的 label 字段 (人工标注), 不是 BERT 文件的 entities。
    """
    preds, labels = [], []
    for ll, bl in zip(llm_lines, bert_lines):
        ll_obj = _safe_parse_json(ll)
        if ll_obj is None:
            preds.append([])
            labels.append([])
            continue
        text = ll_obj.get('text', '')
        models = ll_obj.get(source, []) or ll_obj.get('ner_models', [])
        if beam_idx is not None and isinstance(beam_idx, int) and beam_idx < len(models):
            predict_str = models[beam_idx].get('predict', '')
        else:
            predict_str = ''
        ents = _parse_mark_format_predict(predict_str, text, default_conf=default_conf)
        preds.append([
            {'start_idx': e['start_idx'], 'end_idx': e['end_idx'],
             'type': e['type'], 'entity': e['entity']}
            for e in ents.values()
        ])
        label_text = ll_obj.get('label', '')
        raw_ents = parse_marked_text_with_pos(label_text)
        raw_ents = _correct_entity_positions(raw_ents, text)
        label_ents = [
            {'start_idx': s, 'end_idx': e, 'entity': txt, 'type': t}
            for s, e, txt, t, _ in raw_ents
        ]
        labels.append(label_ents)
    return evaluate(preds, labels)


# ====================================================================
# 4. 数据加载 & 划分
# ====================================================================
def load_data(llm_path, bert_path):
    with open(llm_path, 'r', encoding='utf-8') as f1, \
         open(bert_path, 'r', encoding='utf-8') as f2:
        llm_lines = f1.readlines()
        bert_lines = f2.readlines()
    n = min(len(llm_lines), len(bert_lines))
    return llm_lines[:n], bert_lines[:n]


def split_train_test(llm_lines, bert_lines, test_size=2000, seed=GLOBAL_SEED):
    """按 test_size 切分测试集, 其余为训练集, 用 seed 复现切分结果。"""
    n = len(llm_lines)
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(np.arange(n), size=min(test_size, n), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = [i for i in range(n) if i not in test_idx_set]
    train_llm = [llm_lines[i] for i in train_idx]
    train_bert = [bert_lines[i] for i in train_idx]
    test_llm = [llm_lines[i] for i in sorted(test_idx_set)]
    test_bert = [bert_lines[i] for i in sorted(test_idx_set)]
    return (train_llm, train_bert), (test_llm, test_bert)


def build_samples(llm_lines, bert_lines, source='ner_models', tol=POSITION_TOLERANCE,
                  use_beam=False):
    """对每行生成 (text, candidates, label_set) 三元组, 用于训练门控网络 / 评估。

    use_beam=True:  解析 LLM 时使用 5-beam 投票, 候选中携带 vote_count 等特征。
    ground truth 一律采用 LLM 文件 label 字段 (人工标注)。
    """
    samples = []
    for i, (ll, bl) in enumerate(zip(llm_lines, bert_lines)):
        parsed = parse_llm_line(ll, source=source, use_beam=use_beam)
        if parsed is None:
            continue
        text, entity_dict, label_entities, llm_err = parsed
        if llm_err:
            entity_dict = {}
        raw_ents = parse_marked_text_with_pos(label_entities) if isinstance(label_entities, str) else \
            [(e['start_idx'], e['end_idx'], e['entity'], e['type'], -1) for e in label_entities]
        if not raw_ents and label_entities and isinstance(label_entities[0], dict):
            raw_ents = [(e['start_idx'], e['end_idx'], e['entity'], e['type'], -1) for e in label_entities]
        raw_ents = _correct_entity_positions(raw_ents, text)
        label_entities = [
            {'start_idx': s, 'end_idx': e, 'entity': txt, 'type': t}
            for s, e, txt, t, _ in raw_ents
        ]
        try:
            _, bert_ents = parse_bert_line(bl)
        except Exception:
            bert_ents = []
        cands = build_candidates(text, entity_dict, bert_ents, tol=tol)
        lset = _label_set(label_entities, tol=tol)
        samples.append({
            'text': text,
            'candidates': cands,
            'label_entities': label_entities,
            'label_set': lset,
        })
    return samples


# ====================================================================
# 5. 策略A: 静态权重融合 (Static Weight Fusion)
# ====================================================================
# 思路: 不训练, 在验证集上统计 BERT-only F1 和 LLM-only F1, 然后用两者比值
# 作为权重: α = bert_F1 / (bert_F1 + llm_F1), score = α*bert_conf + (1-α)*llm_conf
# 网格搜索全局阈值 th, 进一步 per-type 调阈值。
def bert_only_predict(samples, llm_lines, bert_lines, source='ner_models'):
    """BERT Only baseline: 仅保留 BERT 预测的实体。"""
    preds_list, labels_list = [], []
    for s in samples:
        bert_pred = [c for c in s['candidates'] if c['bert_present']]
        preds_list.append(bert_pred)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


def llm_only_predict(samples, llm_lines=None, bert_lines=None,
                     source='ner_models', beam_idx=None, conf_th=None):
    """LLM Only baseline: 仅保留 LLM 预测的实体。

    参数:
        beam_idx: int | None
            None  -> 5-beam 并集 (无阈值, 仅按位置去重)  ← ablation
            0..4  -> 仅用第 beam_idx 个 beam 预测    ← 论文主基线 (LLM Top-1)
        conf_th: float | None
            None   -> 不做 conf 过滤
            0.0~1  -> 仅保留 best_beam_conf >= conf_th 的实体
    """
    preds_list, labels_list = [], []
    for s in samples:
        kept = []
        for c in s['candidates']:
            if not c['llm_present']:
                continue
            if beam_idx is not None:
                idxs = c.get('llm_beam_idxs', [int(c.get('best_beam_idx', 0))])
                if beam_idx not in idxs:
                    continue
            if conf_th is not None:
                if c.get('best_beam_conf', c['llm_conf']) < conf_th:
                    continue
            kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                         'type': c['type'], 'entity': c['entity']})
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


def per_type_f1(samples, source='ner_models'):
    """在验证集上按 type 统计 BERT-only 和 LLM-only 的实体级 F1。"""
    bert_tp = defaultdict(int); bert_pred = defaultdict(int); bert_true = defaultdict(int)
    llm_tp = defaultdict(int);  llm_pred = defaultdict(int);  llm_true = defaultdict(int)

    for s in samples:
        lset = s['label_set']
        for c in s['candidates']:
            t = c['type']
            k = (c['start_idx'], c['entity'], t)
            if c['bert_present']:
                bert_pred[t] += 1
                bert_true[t] += 1
                if k in lset:
                    bert_tp[t] += 1
            if c['llm_present']:
                llm_pred[t] += 1
                llm_true[t] += 1
                if k in lset:
                    llm_tp[t] += 1

    def _f1(tp, pred, true):
        p = tp / pred if pred else 0.0
        r = tp / true if true else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    bert_f1 = {t: _f1(bert_tp[t], bert_pred[t], bert_true[t]) for t in ALL_TYPES}
    llm_f1 = {t: _f1(llm_tp[t], llm_pred[t], llm_true[t]) for t in ALL_TYPES}
    return bert_f1, llm_f1


def _consensus_stats(samples):
    """统计: 共识 (BERT+LLM 同预测) 的精确率 / 单独预测的精确率, 是策略 A 共识加成的依据。"""
    both_p = both_t = 0
    bert_p = bert_t = 0
    llm_p = llm_t = 0
    for s in samples:
        lset = s['label_set']
        for c in s['candidates']:
            k = (c['start_idx'], c['entity'], c['type'])
            hit = (k in lset)
            if c['bert_present'] and c['llm_present']:
                both_t += 1; both_p += int(hit)
            elif c['bert_present']:
                bert_t += 1; bert_p += int(hit)
            elif c['llm_present']:
                llm_t += 1; llm_p += int(hit)
    return {
        'both_prec':  both_p / both_t if both_t else 0.0,
        'bert_prec':  bert_p / bert_t if bert_t else 0.0,
        'llm_prec':   llm_p  / llm_t  if llm_t  else 0.0,
        'both_n':     both_t,
        'bert_only_n': bert_t,
        'llm_only_n':  llm_t,
    }


def static_fusion_predict(samples, type_alphas, type_thresholds,
                          consensus_bonus=0.10, consensus_th_mult=0.85,
                          bert_only_factor=1.0, llm_only_factor=1.0,
                          vote_bonus_coef=0.05):
    """静态权重融合 (3-case 评分)。

    核心思想: 共识实体的精度远高于单源预测, 应当显著加分并降低阈值。
        both-present: score = α*bert + (1-α)*llm + bonus + vote_bonus_coef * log(vc)/log(5),  th *= consensus_th_mult
        bert-only:    score = bert_only_factor * bert_conf,  th 不变
        llm-only:     score = llm_only_factor * llm_conf,   th 不变
    """
    preds_list, labels_list = [], []
    for s in samples:
        kept = []
        for c in s['candidates']:
            t = c['type']
            alpha = type_alphas.get(t, 0.5)
            th = type_thresholds.get(t, 0.5)
            if c['bert_present'] and c['llm_present']:
                vc = max(c.get('vote_count', 1.0), 1.0)
                vote_bonus = vote_bonus_coef * math.log(vc) / math.log(5)
                score = (alpha * c['bert_conf'] + (1 - alpha) * c['llm_conf']
                         + consensus_bonus + vote_bonus)
                th_eff = th * consensus_th_mult
            elif c['bert_present']:
                score = bert_only_factor * c['bert_conf']
                th_eff = th
            else:  # llm_only
                score = llm_only_factor * c['llm_conf']
                th_eff = th
            if score >= th_eff:
                kept.append({
                    'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                    'type': t, 'entity': c['entity'],
                })
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


def static_fusion_predict_metrics(samples, type_alphas, type_thresholds,
                                    consensus_bonus=0.10, consensus_th_mult=0.85,
                                    bert_only_factor=1.0, llm_only_factor=1.0,
                                    vote_bonus_coef=0.05):
    """static_fusion_predict 的快速版本: 返回 (tps, pns, tns) np.array, 不构 list of dict。

    利用 s['label_set'] (已是 set) 算 hit, 比 evaluate 快 5-10x。
    用途: run_static_fusion 全局网格 6000 次 evaluate 加速。
    """
    n = len(samples)
    tps = np.zeros(n, dtype=np.int64)
    pns = np.zeros(n, dtype=np.int64)
    tns = np.empty(n, dtype=np.int64)
    for i, s in enumerate(samples):
        lset = s['label_set']  # 已 tol 膨胀, 仅用于 O(1) 命中检查
        n_keep = 0; n_hit = 0
        for c in s['candidates']:
            t = c['type']
            alpha = type_alphas.get(t, 0.5)
            th = type_thresholds.get(t, 0.5)
            if c['bert_present'] and c['llm_present']:
                vc = max(c.get('vote_count', 1.0), 1.0)
                vote_bonus = vote_bonus_coef * math.log(vc) / math.log(5)
                score = (alpha * c['bert_conf'] + (1 - alpha) * c['llm_conf']
                         + consensus_bonus + vote_bonus)
                th_eff = th * consensus_th_mult
            elif c['bert_present']:
                score = bert_only_factor * c['bert_conf']
                th_eff = th
            else:  # llm_only
                score = llm_only_factor * c['llm_conf']
                th_eff = th
            if score >= th_eff:
                n_keep += 1
                key = (c['start_idx'], c['entity'], t)
                if key in lset:
                    n_hit += 1
        tps[i] = n_hit
        pns[i] = n_keep
        # 注意: label_set 已 tol 膨胀, 不能直接 len; 用真 label 数 (与 evaluate 对齐)
        tns[i] = len(s['label_entities'])
    return tps, pns, tns


def _f1_from_arr(tps, pns, tns):
    """从 (tps, pns, tns) 数组整体算 P/R/F1 — 给 fast 版 evaluate 用"""
    tp = int(tps.sum())
    pn = int(pns.sum())
    tn = int(tns.sum())
    p = tp / pn if pn else 0.0
    r = tp / tn if tn else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def run_static_fusion(train_samples, test_samples):
    """策略 A: 静态权重融合 (3-case, 含共识加成)"""
    print("\n" + "=" * 70)
    print("【策略 A】Static Weight Fusion (静态权重融合 · 3-case)")
    print("=" * 70)
    print("思路: 共识实体的精度远高于单源预测。")
    print("      - both-present: score = α*bert + (1-α)*llm + bonus, 阈值 × mult")
    print("      - bert-only:    score = bert_conf")
    print("      - llm-only:     score = llm_conf")
    print("      网格搜: α (per-type) + bonus / mult / factor (全局) + 阈值 (per-type)")
    print("-" * 70)

    # 1) 验证集 per-type F1 (用于 α) + 共识诊断
    bert_f1, llm_f1 = per_type_f1(train_samples)
    cs = _consensus_stats(train_samples)
    print("  [1] 验证集单模型 F1 + 共识诊断:")
    print(f"      {'type':<6} {'BERT F1':>9} {'LLM F1':>9} {'α':>8}")
    for t in ALL_TYPES:
        b, l = bert_f1[t], llm_f1[t]
        denom = b + l
        a = b / denom if denom > 0 else 0.5
        print(f"      {t:<6} {b:>9.4f} {l:>9.4f} {a:>8.4f}")
    print(f"      共识 both_prec  = {cs['both_prec']:.4f}  (n={cs['both_n']})")
    print(f"      bert_only_prec  = {cs['bert_prec']:.4f}  (n={cs['bert_only_n']})")
    print(f"      llm_only_prec   = {cs['llm_prec']:.4f}  (n={cs['llm_only_n']})")

    # 2) 搜索全局超参: consensus_bonus, consensus_th_mult, bert_only_factor, llm_only_factor
    #    + vote_bonus_coef (5-beam 票数加成)
    #    每种类型 α 按 F1 比例固定; 阈值按 (per-type) 网格搜
    #    目标: F1 - 0.02 * |P - R|  (P/R 平衡, 防过拟合)
    print("\n  [2] 全局超参网格搜索 (在验证集上, 目标 F1 - 0.02·|P-R|):")
    best = {'f1': 0.0, 'score': -1.0}
    type_alphas = {t: (bert_f1[t] / (bert_f1[t] + llm_f1[t])
                       if (bert_f1[t] + llm_f1[t]) > 0 else 0.5) for t in ALL_TYPES}
    # 加速: 走 static_fusion_predict_metrics (fast 版, 直接给 tps/pns/tns)
    # 搜参空间: bonus×mult×bf×lf×vbc×th, Step 1.5 已单独搜过 β→vbc, 此处固定 vbc=0 (最优),
    #          bonus/bf/lf 收窄到经验有效范围 → 1080 次 (原 7200 次, 提速 6.7×)
    _bonus_grid = [0.05, 0.15, 0.25]
    _mult_grid = [0.7, 0.85, 1.0]
    _bf_grid = [0.9, 1.0]
    _lf_grid = [0.7, 0.85, 1.0]
    _vbc_grid = [0.00]  # Step 1.5 已搜过 β, 此处 vbc 固定 0
    _th_grid = [0.30, 0.35, 0.40, 0.45, 0.50]
    _total = len(_bonus_grid) * len(_mult_grid) * len(_bf_grid) * len(_lf_grid) * len(_vbc_grid) * len(_th_grid)
    _t0 = time.time()
    _cnt = 0
    for bonus in _bonus_grid:
        for mult in _mult_grid:
            for bf in _bf_grid:
                for lf in _lf_grid:
                    for vbc in _vbc_grid:
                        for th in _th_grid:
                            tps, pns, tns = static_fusion_predict_metrics(
                                train_samples, type_alphas,
                                {t: th for t in ALL_TYPES},
                                consensus_bonus=bonus, consensus_th_mult=mult,
                                bert_only_factor=bf, llm_only_factor=lf,
                                vote_bonus_coef=vbc)
                            p, r, f1 = _f1_from_arr(tps, pns, tns)
                            score = f1 - 0.02 * abs(p - r)
                            if score > best['score']:
                                best = {'f1': f1, 'p': p, 'r': r, 'score': score,
                                        'bonus': bonus, 'mult': mult,
                                        'bf': bf, 'lf': lf, 'th': th, 'vbc': vbc}
                            _cnt += 1
                            if _cnt % 100 == 0 or _cnt == _total:
                                _elapsed = time.time() - _t0
                                _eta = _elapsed / _cnt * (_total - _cnt)
                                print(f"      [进度] {_cnt}/{_total}  "
                                      f"已用 {_elapsed:.1f}s, 预计剩余 {_eta:.1f}s, "
                                      f"当前最优 F1={best['f1']:.4f}")
    print(f"  -> 全局最佳: bonus={best['bonus']}, mult={best['mult']}, "
          f"bf={best['bf']}, lf={best['lf']}, th={best['th']}, "
          f"vbc={best['vbc']}, F1={best['f1']:.4f} (P={best['p']:.4f}, R={best['r']:.4f})")

    # 3) per-type 阈值微调 (fast 版)
    print("\n  [3] per-type 阈值微调:")
    best_type_th = {t: best['th'] for t in ALL_TYPES}
    for etype in ALL_TYPES:
        best_t, best_f_t = best['th'], 0.0
        for th in np.arange(0.10, 0.85, 0.05):
            test_th = dict(best_type_th)
            test_th[etype] = float(th)
            tps, pns, tns = static_fusion_predict_metrics(
                train_samples, type_alphas, test_th,
                consensus_bonus=best['bonus'], consensus_th_mult=best['mult'],
                bert_only_factor=best['bf'], llm_only_factor=best['lf'],
                vote_bonus_coef=best['vbc'])
            _, _, f1 = _f1_from_arr(tps, pns, tns)
            if f1 > best_f_t:
                best_f_t, best_t = f1, float(th)
        best_type_th[etype] = best_t
        print(f"      {etype}: th={best_t:.2f}, F1={best_f_t:.4f}")

    # 4) 验证集最终
    print("\n  [4] 验证集最终结果:")
    val_preds, val_labels = static_fusion_predict(
        train_samples, type_alphas, best_type_th,
        consensus_bonus=best['bonus'], consensus_th_mult=best['mult'],
        bert_only_factor=best['bf'], llm_only_factor=best['lf'],
        vote_bonus_coef=best['vbc'])
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels)
    print_metrics("Static-Weight (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    # 5) 测试集
    test_preds, test_labels = static_fusion_predict(
        test_samples, type_alphas, best_type_th,
        consensus_bonus=best['bonus'], consensus_th_mult=best['mult'],
        bert_only_factor=best['bf'], llm_only_factor=best['lf'],
        vote_bonus_coef=best['vbc'])
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels)
    print_metrics("Static-Weight (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)
    # 返回 (F1_val, F1_test, type_alphas, type_thresholds, 超参 dict)
    hyper = {
        'bonus': best['bonus'], 'mult': best['mult'],
        'bf': best['bf'], 'lf': best['lf'], 'vbc': best['vbc'],
    }
    return val_f1, test_f1, type_alphas, best_type_th, hyper


# ====================================================================
# 5b. 策略 A'' (Vote-Aware 硬规则版): 把 beam 票数显式编入阈值
# ====================================================================
# 在 A' 基础上增加两个全局参数:
#   min_vote_both: 共识 (BERT+LLM 共同) 至少需要几 beam 同意
#   min_vote_llm : 单独 LLM 至少需要几 beam 同意 (单独 LLM 容易幻觉, 默认要 5 票)
# 直观解释:
#   - 5 票单独 LLM = 5 个 beam 都猜这个实体, 但 BERT 没猜 → 可能是 LLM 学到而 BERT 没学到的
#   - 1 票共识 = 1 个 beam 同意 + BERT 同意 → 弱信号, 可拒
#   - 5 票共识 = 全员同意 → 强信号, 必留

def _make_uniform_type_th(con_th, b_th, l_th):
    """生成 type_th dict: {type: (con_th, b_th, l_th)} 全部类型用同一组阈值。"""
    return {t: (con_th, b_th, l_th) for t in ALL_TYPES}


def static_fusion_voteaware_predict(samples, type_th, min_vote_both=1, min_vote_llm=5):
    """Vote-Aware 硬规则融合。
    type_th[t] = (con_th, b_th, l_th)
    规则:
        共识:        vote_count >= min_vote_both AND bert_conf >= con_th[t]
        单独 BERT:   bert_conf >= b_th[t]
        单独 LLM:    vote_count >= min_vote_llm  AND llm_conf  >= l_th[t]
    """
    preds_list, labels_list = [], []
    for s in samples:
        kept = []
        for c in s['candidates']:
            t = c['type']
            con_th, b_th, l_th = type_th.get(t, (0.0, 0.9, 0.9))
            vc = int(c.get('vote_count', 1))
            if c['bert_present'] and c['llm_present']:
                if vc >= min_vote_both and c['bert_conf'] >= con_th:
                    kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                 'type': t, 'entity': c['entity']})
            elif c['bert_present']:
                if c['bert_conf'] >= b_th:
                    kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                 'type': t, 'entity': c['entity']})
            else:  # llm_only
                if vc >= min_vote_llm and c['llm_conf'] >= l_th:
                    kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                 'type': t, 'entity': c['entity']})
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


def run_static_fusion_voteaware(train_samples, test_samples):
    """策略 A'': Vote-Aware Hard-Rule Static Fusion"""
    print("\n" + "=" * 70)
    print("【策略 A''】Vote-Aware Hard-Rule Static Fusion (硬规则 + 票数门)")
    print("=" * 70)
    print("思路: 在 A' 基础上把 beam 票数显式编入判定:")
    print("      - 共识: vote_count >= min_vote_both AND bert_conf >= con_th")
    print("      - 单独 LLM: vote_count >= min_vote_llm AND llm_conf >= l_th")
    print("      单独 LLM 经常是幻觉, 默认要 5/5 票才考虑。")
    print("-" * 70)

    # 1) 全局网格: min_vote_both (共识最少票) × min_vote_llm (单独 LLM 最少票)
    #            × bsolo (BERT solo 阈值) × lsolo (LLM solo 阈值, con=0 固定)
    print("  [1] 全局网格 (min_vote_both × min_vote_llm × bsolo × lsolo):")
    best = {'f1': 0.0}
    for mvb in [1, 2, 3, 4, 5]:                       # 共识最少票
        for mvl in [3, 4, 5]:                         # 单独 LLM 最少票
            for bsolo in [0.80, 0.85, 0.90, 0.95]:
                for lsolo in [0.80, 0.85, 0.90, 0.95]:
                    type_th = _make_uniform_type_th(0.0, bsolo, lsolo)
                    preds, labels = static_fusion_voteaware_predict(
                        train_samples, type_th, mvb, mvl)
                    _, _, f1, _, _, _ = evaluate(preds, labels)
                    if f1 > best['f1']:
                        best = {'f1': f1, 'mvb': mvb, 'mvl': mvl,
                                'bsolo': bsolo, 'lsolo': lsolo}
    print(f"  -> 全局最佳: min_vote_both={best['mvb']}, min_vote_llm={best['mvl']}, "
          f"bsolo={best['bsolo']}, lsolo={best['lsolo']}, F1={best['f1']:.4f}")

    # 2) per-type consensus_th 微调 (影响最大, 单独阈值共用全局值)
    print("\n  [2] per-type consensus_th 微调:")
    type_th = _make_uniform_type_th(0.0, best['bsolo'], best['lsolo'])
    for etype in ALL_TYPES:
        best_con, best_f_t = 0.0, 0.0
        for con in np.arange(0.0, 0.55, 0.02):
            test_th = dict(type_th)
            test_th[etype] = (float(con), best['bsolo'], best['lsolo'])
            preds, labels = static_fusion_voteaware_predict(
                train_samples, test_th, best['mvb'], best['mvl'])
            _, _, f1, _, _, _ = evaluate(preds, labels)
            if f1 > best_f_t:
                best_f_t, best_con = f1, float(con)
        type_th[etype] = (best_con, best['bsolo'], best['lsolo'])
        print(f"      {etype}: con_th={best_con:.2f}, F1={best_f_t:.4f}")

    # 3) per-type 单独阈值微调 (围绕全局值)
    print("\n  [3] per-type bert_solo_th / llm_solo_th 微调:")
    for etype in ALL_TYPES:
        cur_con, _, _ = type_th[etype]
        best_b, best_l, best_f_t = best['bsolo'], best['lsolo'], 0.0
        for bsolo in np.arange(max(0.30, best['bsolo'] - 0.20),
                                min(0.98, best['bsolo'] + 0.20) + 1e-9, 0.05):
            for lsolo in np.arange(max(0.30, best['lsolo'] - 0.20),
                                    min(0.98, best['lsolo'] + 0.20) + 1e-9, 0.05):
                test_th = dict(type_th)
                test_th[etype] = (cur_con, float(bsolo), float(lsolo))
                preds, labels = static_fusion_voteaware_predict(
                    train_samples, test_th, best['mvb'], best['mvl'])
                _, _, f1, _, _, _ = evaluate(preds, labels)
                if f1 > best_f_t:
                    best_f_t, best_b, best_l = f1, float(bsolo), float(lsolo)
        type_th[etype] = (cur_con, best_b, best_l)
        print(f"      {etype}: con={cur_con:.2f}, bsolo={best_b:.2f}, lsolo={best_l:.2f}, F1={best_f_t:.4f}")

    # 4) per-type 二次 min_vote 调优 (围绕全局)
    print("\n  [4] per-type min_vote_both 微调 (单独 LLM 票数共用全局):")
    for etype in ALL_TYPES:
        cur_con, cur_b, cur_l = type_th[etype]
        best_mvb, best_f_t = best['mvb'], 0.0
        for mvb in [1, 2, 3, 4, 5]:
            test_th = dict(type_th)
            test_th[etype] = (cur_con, cur_b, cur_l)
            preds, labels = static_fusion_voteaware_predict(
                train_samples, test_th, mvb, best['mvl'])
            _, _, f1, _, _, _ = evaluate(preds, labels)
            if f1 > best_f_t:
                best_f_t, best_mvb = f1, mvb
        type_th[etype] = (cur_con, cur_b, cur_l)
        # 注意: 实际 min_vote_both 是全局的, 此处只是验证全局值的稳定性
    # 全部类型共用最佳 mvb / mvl
    mvb_final = best['mvb']
    mvl_final = best['mvl']
    print(f"      全部类型使用全局: min_vote_both={mvb_final}, min_vote_llm={mvl_final}")

    # 5) 验证集
    print("\n  [5] 验证集最终结果:")
    val_preds, val_labels = static_fusion_voteaware_predict(
        train_samples, type_th, mvb_final, mvl_final)
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels)
    print_metrics("Vote-Aware Hard-Rule (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    # 6) 测试集
    test_preds, test_labels = static_fusion_voteaware_predict(
        test_samples, type_th, mvb_final, mvl_final)
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels)
    print_metrics("Vote-Aware Hard-Rule (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)
    return val_f1, test_f1, type_th, mvb_final, mvl_final


# ====================================================================
# 6. 策略B: 门控网络融合 (Gating Network Fusion)
#   - 网络输出 (w_bert, w_llm) 两路 sigmoid 权重
#   - 融合分 = w_bert * bert_conf + w_llm * llm_conf + consensus_bonus * both_present
#   - 训练目标: 让"实体正确"时 combined_score 大, 错误时小 (BCE on combined)
#   - 共识特征 (both_present) 进入网络, 也作为预测时显式加成
#   - 网络会自动学会: 共识 → w_b + w_l 都高; 单独 → 偏向高置信度的那一侧
# ====================================================================

class GatingNetwork(nn.Module):
    """输出两路权重 (w_bert, w_llm), 网络可学习"何时信任谁"
    n_feats: 输入特征数。完整=16 (含 5-beam 特有 4 个), drop_beam=12 (只用通用特征)

    架构分两种:
      - 5b 模式 (n_feats=16): 2×hidden=64 + 3 头 (w_bert/w_llm/bonus) — 与原版一致
      - b1 模式 (n_feats=12): 1×hidden=32 + 1 头, 直接输出单 logit — 对齐 gating_network_dp_mark_v3.py
                              配合 BCE + 0.15*MSE(calibrated_target) 蒸馏损失, 缓解 val/test 过拟合
    """
    def __init__(self, num_types=NUM_TYPES, type_emb_dim=8, hidden_dim=64, n_feats=16):
        super().__init__()
        self.n_feats = n_feats
        self.drop_beam_features = (n_feats == 12)  # 模型自己记住, 推理时 gating_predict 会读
        self.type_emb = nn.Embedding(num_types, type_emb_dim)
        if n_feats == 12:
            # b1 模式: 小容量 + 单头, 避免在 Beam-1 小样本上过拟合
            self.trunk = nn.Sequential(
                nn.Linear(12 + type_emb_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.head_score = nn.Linear(32, 1)
        else:
            # 5b 模式: 完整 3 头, 与原版一致
            in_dim = n_feats + type_emb_dim
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            # 头1: 输出 w_bert
            self.head_wb = nn.Linear(hidden_dim, 1)
            # 头2: 输出 w_llm
            self.head_wl = nn.Linear(hidden_dim, 1)
            # 头3: 共识加成 (sigmoid 输出后乘以 both_pres)
            self.head_bonus = nn.Linear(hidden_dim, 1)

    def forward(self, llm_conf, bert_conf, llm_pres, bert_pres,
                conf_diff, both_pres, llm_only, bert_only,
                n_cands, bert_rank, llm_rank, static_score,
                type_ids, *beam_feats):
        """参数顺序: 12 base + type_ids + N beam_feats (0 或 4)
        完整模式 (drop=False): beam_feats 收到 4 个 tensor
        drop 模式 (drop=True):  beam_feats 是空 tuple, 跳过
        """
        type_e = self.type_emb(type_ids)
        base = [llm_conf, bert_conf, llm_pres, bert_pres,
                conf_diff, both_pres, llm_only, bert_only,
                n_cands, bert_rank, llm_rank, static_score]
        x = torch.stack(base + list(beam_feats), dim=1)
        x = torch.cat([x, type_e], dim=1)
        h = self.trunk(x)
        if self.n_feats == 12:
            # b1 模式: 直接返回单 logit (配合 sigmoid 得 0~1 分)
            return self.head_score(h).squeeze(-1)
        # 5b 模式: 3 头
        w_bert_logit = self.head_wb(h).squeeze(-1)
        w_llm_logit  = self.head_wl(h).squeeze(-1)
        bonus_logit  = self.head_bonus(h).squeeze(-1)
        return w_bert_logit, w_llm_logit, bonus_logit


def _candidates_to_features(samples, drop_beam_features=False):
    """把 samples 拆成 (cand_features, type_ids, group_ids, labels) 列表
    drop_beam_features=True:  不喂 5-beam 特有特征 (vote_count/llm_avg_conf/llm_max_conf/best_beam_idx),
                              强制门控网络只用通用特征 (B_b1 消融用)。
    """
    feats, type_ids, group_ids, labels = [], [], [], []
    for gi, s in enumerate(samples):
        lset = s['label_set']
        n_c = float(len(s['candidates']))
        bert_cands = [(i, c) for i, c in enumerate(s['candidates']) if c['bert_present']]
        bert_cands.sort(key=lambda x: x[1]['bert_conf'], reverse=True)
        bert_rank = {i: r / max(len(bert_cands) - 1, 1) if len(bert_cands) > 1 else 0.5
                     for r, (i, _) in enumerate(bert_cands)}
        llm_cands = [(i, c) for i, c in enumerate(s['candidates']) if c['llm_present']]
        llm_cands.sort(key=lambda x: x[1]['llm_conf'], reverse=True)
        llm_rank = {i: r / max(len(llm_cands) - 1, 1) if len(llm_cands) > 1 else 0.5
                    for r, (i, _) in enumerate(llm_cands)}
        for i, c in enumerate(s['candidates']):
            t = c['type']
            if t not in TYPE2IDX:
                continue
            k = (c['start_idx'], c['entity'], t)
            if drop_beam_features:
                # B_b1 消融: 完全不喂 5-beam 特有特征, 强制只用通用特征
                # 多算 calibrated_target (0.5*llm + 0.5*bert) 给蒸馏损失用 — 对齐 v3
                feat = {
                    'llm_conf': c['llm_conf'], 'bert_conf': c['bert_conf'],
                    'llm_present': float(c['llm_present']), 'bert_present': float(c['bert_present']),
                    'conf_diff': c['bert_conf'] - c['llm_conf'],
                    'both_pres': float(c['llm_present'] and c['bert_present']),
                    'llm_only':  float(c['llm_present'] and not c['bert_present']),
                    'bert_only': float(c['bert_present'] and not c['llm_present']),
                    'n_cands': n_c,
                    'bert_rank': bert_rank.get(i, 0.5),
                    'llm_rank': llm_rank.get(i, 0.5),
                    'static_score': 0.5 * c['bert_conf'] + 0.5 * c['llm_conf'],
                    'calibrated_target': 0.5 * c['llm_conf'] + 0.5 * c['bert_conf'],
                    'type_id': TYPE2IDX[t],
                }
            else:
                # 候选若有 beam 特征则取, 否则填默认值 (0.0 = 单 beam 视角)
                vc = float(c.get('vote_count', 0.0))
                la = float(c.get('llm_avg_conf', 0.0))
                lm = float(c.get('llm_max_conf', 0.0))
                feat = {
                    'llm_conf': c['llm_conf'], 'bert_conf': c['bert_conf'],
                    'llm_present': float(c['llm_present']), 'bert_present': float(c['bert_present']),
                    'conf_diff': c['bert_conf'] - c['llm_conf'],
                    'both_pres': float(c['llm_present'] and c['bert_present']),
                    'llm_only':  float(c['llm_present'] and not c['bert_present']),
                    'bert_only': float(c['bert_present'] and not c['llm_present']),
                    'n_cands': n_c,
                    'bert_rank': bert_rank.get(i, 0.5),
                    'llm_rank': llm_rank.get(i, 0.5),
                    'static_score': 0.5 * c['bert_conf'] + 0.5 * c['llm_conf'],
                    'vote_count': vc,
                    'llm_avg_conf': la,
                    'llm_max_conf': lm,
                    'best_beam_idx': float(c.get('best_beam_idx', 0.0)),
                    'type_id': TYPE2IDX[t],
                }
            feats.append(feat)
            type_ids.append(TYPE2IDX[t])
            group_ids.append(gi)
            labels.append(1 if k in lset else 0)
    return feats, type_ids, group_ids, labels


def _to_tensors(feats, type_ids, group_ids, labels, drop_beam_features=False):
    """返回顺序: 12 base + type_ids + (4 beam?) + group_ids + (calib?) + label
    完整模式 (drop=False): 12 + 1 + 4 + 1 + 0 + 1 = 19 个
    drop 模式 (drop=True):   12 + 1 + 0 + 1 + 1 + 1 = 16 个  (calib=0.5*llm+0.5*bert 给蒸馏用)
    forward 签名: 12 base + type_ids + 4 beam (顺序一一对应, model(*xs) 用位置参数传)
    """
    if not feats:
        n_base = 12
        n_beam = 0 if drop_beam_features else 4
        empty_f = torch.empty(0, dtype=torch.float32)
        empty_l = torch.empty(0, dtype=torch.long)
        # 顺序: n_base 个 float + 1 long (type) + n_beam 个 float + 1 long (group) + (1 float calib) + 1 float (label)
        out = ((empty_f,) * n_base + (empty_l,)
               + (empty_f,) * n_beam + (empty_l,))
        if drop_beam_features:
            out = out + (empty_f,)
        out = out + (empty_f,)
        return out
    base_keys = ['llm_conf', 'bert_conf', 'llm_present', 'bert_present',
                 'conf_diff', 'both_pres', 'llm_only', 'bert_only',
                 'n_cands', 'bert_rank', 'llm_rank', 'static_score']
    beam_keys = ['vote_count', 'llm_avg_conf', 'llm_max_conf', 'best_beam_idx']
    tensors = [torch.tensor([f[k] for f in feats], dtype=torch.float32) for k in base_keys]
    tensors.append(torch.tensor(type_ids, dtype=torch.long))   # type_ids 紧跟 12 base 之后
    if not drop_beam_features:
        tensors += [torch.tensor([f[k] for f in feats], dtype=torch.float32) for k in beam_keys]
    tensors.append(torch.tensor(group_ids, dtype=torch.long))
    if drop_beam_features:
        # b1 模式: 追加 calibrated_target (蒸馏软标签), 在 group 之后 label 之前
        tensors.append(torch.tensor([f['calibrated_target'] for f in feats], dtype=torch.float32))
    tensors.append(torch.tensor(labels, dtype=torch.float32))
    return tuple(tensors)


def _grouped_ranking_loss(scores, labels, group_ids, margin=0.05):
    pos = labels > 0.5
    neg = labels < 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return torch.tensor(0.0, device=scores.device)
    same = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
    pairs = pos.unsqueeze(1) & neg.unsqueeze(0)
    valid = same & pairs
    n_valid = valid.float().sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=scores.device)
    diff = scores.unsqueeze(0) - scores.unsqueeze(1) + margin
    return (torch.clamp(diff, min=0) * valid.float()).sum() / n_valid


def _gating_combined(w_bert, w_llm, bonus_logit, bert_conf, llm_conf, both_pres):
    """把网络 logits 拼成标量 '实体正确分' (用于训练和预测)"""
    wb = torch.sigmoid(w_bert)
    wl = torch.sigmoid(w_llm)
    b  = torch.sigmoid(bonus_logit)
    # 共识加成只在 both_present=1 时生效
    bonus_term = b * both_pres * 0.30
    return wb * bert_conf + wl * llm_conf + bonus_term, wb, wl, b


def _bce_loss_with_ranking(combined, labels, group_ids, margin=0.05):
    bce = nn.functional.binary_cross_entropy_with_logits(combined, labels)
    rank = _grouped_ranking_loss(combined, labels, group_ids, margin=margin)
    # ranking 主导 (0.3), BCE 辅助 (1.0) → 缓解保守阈值
    return bce + 0.3 * rank


def train_gating(model, train_loader, valid_loader, epochs=50, lr=2e-3, patience=10,
                 save_dir='saved_models_clean', model_tag='5b'):
    """训练门控网络; 评估指标用 combined 的 BCE + ranking
    model_tag:  '5b' (5-beam 特征, 16 维) 或 'b1' (Beam-1 only, 12 维) — 决定 best 模型文件名
                避免两个消融共用同一文件名互相覆盖
    b1 模式: lr=5e-4 / wd=1e-3 / patience=15, 损失 = BCE + 0.15·MSE(calibrated_target)
             对齐 gating_network_dp_mark_v3.py
    """
    os.makedirs(save_dir, exist_ok=True)
    is_b1 = (model_tag == 'b1')
    # b1 模式用更稳的超参 (小 lr + 小 wd + 长 patience) — 与 v3 一致
    # 注: b1 的 lr 来自调用方传参 (caller 显式传 5e-4), wd/patience 强制覆盖, 避免老 caller 传 5 时被忽略
    if is_b1:
        if lr is None:
            lr = 5e-4
        wd = 1e-3
        patience = 15
        loss_name = "BCE + 0.15*MSE(calib)"
    else:
        wd = 0.01
        loss_name = "BCE + 0.3*rank"
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss, trigger = float('inf'), 0
    # 按 model_tag 区分: 5b / b1 各存一份, 避免互相覆盖
    best_path = os.path.join(save_dir, f'best_gating_{model_tag}.pth')
    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        if is_b1:
            # b1 loader 格式: *xs, gid, calib, y
            for *xs, gid, calib, y in train_loader:
                optimizer.zero_grad()
                score_logit = model(*xs)            # 单 logit
                prob = torch.sigmoid(score_logit)
                bce = nn.functional.binary_cross_entropy_with_logits(score_logit, y)
                mse = nn.functional.mse_loss(prob, calib)
                loss = bce + 0.15 * mse
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
        else:
            # 5b loader 格式: *xs, gid, y
            for *xs, gid, y in train_loader:
                optimizer.zero_grad()
                wb_log, wl_log, bonus_log = model(*xs)
                combined, _, _, _ = _gating_combined(
                    wb_log, wl_log, bonus_log,
                    xs[1],  # bert_conf
                    xs[0],  # llm_conf
                    xs[5],  # both_pres
                )
                loss = _bce_loss_with_ranking(combined, y, gid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
        scheduler.step()
        model.eval()
        v_loss = 0
        with torch.no_grad():
            if is_b1:
                for *xs, gid, calib, y in valid_loader:
                    score_logit = model(*xs)
                    prob = torch.sigmoid(score_logit)
                    bce = nn.functional.binary_cross_entropy_with_logits(score_logit, y)
                    mse = nn.functional.mse_loss(prob, calib)
                    loss = bce + 0.15 * mse
                    v_loss += loss.item()
            else:
                for *xs, gid, y in valid_loader:
                    wb_log, wl_log, bonus_log = model(*xs)
                    combined, _, _, _ = _gating_combined(
                        wb_log, wl_log, bonus_log, xs[1], xs[0], xs[5])
                    loss = _bce_loss_with_ranking(combined, y, gid)
                    v_loss += loss.item()
        v_loss /= max(len(valid_loader), 1)
        print(f"    Epoch {ep:>2d}/{epochs}  train_loss={total / max(len(train_loader), 1):.4f}  val_loss={v_loss:.4f}  [{loss_name}]")
        if v_loss < best_val_loss:
            best_val_loss, trigger = v_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            trigger += 1
            if trigger >= patience:
                print(f"    Early stopping at epoch {ep}.")
                break
    model.load_state_dict(torch.load(best_path, map_location='cpu'))
    return model

def gating_predict(samples, model, threshold):
    """用训练好的门控网络预测。
    分数 = sigmoid(w_bert)*bert_conf + sigmoid(w_llm)*llm_conf + sigmoid(bonus)*both_pres*0.30
    threshold: float (全局) 或 {type: float} (per-type)
    注: model.drop_beam_features (训练时设置) 决定推理时是否 drop 5-beam 特征
    """
    drop = getattr(model, 'drop_beam_features', False)
    model.eval()
    # 一次批量推理 → 拿所有样本的所有候选分数, 避免循环里逐样本调 model
    scores_per_sample = _gating_batch_scores(samples, model, drop)
    preds_list, labels_list = [], []
    for s, sc in zip(samples, scores_per_sample):
        kept = []
        for i, c in enumerate(s['candidates']):
            t = c['type']
            if t not in TYPE2IDX:
                continue
            th = threshold.get(t, 0.5) if isinstance(threshold, dict) else threshold
            if i < len(sc) and float(sc[i]) >= th:
                kept.append({
                    'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                    'type': t, 'entity': c['entity'],
                })
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


def _gating_batch_scores(samples, model, drop):
    """一次性算出 samples 中所有候选的 combined score (向量化推理)。
    返回: list[np.array], 第 s 个数组是 samples[s] 的所有候选的 score。
    网格搜索时复用: 1 次批量推理, N 次阈值扫描 (N=40 全局 + 90 per-type)。
    b1 模式: model 输出单 logit, 得分 = sigmoid(logit)
    5b 模式: model 输出 3 头, 得分 = wb*bert + wl*llm + bonus*both*0.30
    """
    all_feats, all_tids, boundaries = [], [], []
    for s in samples:
        feats, tids, _, _ = _candidates_to_features([s], drop_beam_features=drop)
        start = len(all_feats)
        all_feats.extend(feats)
        all_tids.extend(tids)
        boundaries.append((start, len(all_feats)))
    if not all_feats:
        return [np.array([], dtype=np.float32) for _ in samples]
    # 末尾 2 元素是 (gid, label) 或 3 元素 (gid, calib, label), 但本函数只喂 0 标签, 与训练 loader 不一致
    # 这里直接重新 build 一个无 label 的张量组, 避免歧义
    n_base = 12
    n_beam = 0 if drop else 4
    base_keys = ['llm_conf', 'bert_conf', 'llm_present', 'bert_present',
                 'conf_diff', 'both_pres', 'llm_only', 'bert_only',
                 'n_cands', 'bert_rank', 'llm_rank', 'static_score']
    beam_keys = ['vote_count', 'llm_avg_conf', 'llm_max_conf', 'best_beam_idx']
    x = [torch.tensor([f[k] for f in all_feats], dtype=torch.float32) for k in base_keys]
    x.append(torch.tensor(all_tids, dtype=torch.long))
    if not drop:
        x += [torch.tensor([f[k] for f in all_feats], dtype=torch.float32) for k in beam_keys]
    with torch.no_grad():
        out = model(*x)
        if model.n_feats == 12:
            # b1 模式: 单 logit → sigmoid
            scores = torch.sigmoid(out).cpu().numpy()
        else:
            # 5b 模式: 3 头 → combined
            wb_log, wl_log, bonus_log = out
            combined, _, _, _ = _gating_combined(wb_log, wl_log, bonus_log, x[1], x[0], x[5])
            scores = combined.cpu().numpy()
    return [scores[start:end].astype(np.float32, copy=False)
            for (start, end) in boundaries]


def _filter_by_threshold(samples, scores_per_sample, threshold):
    """基于预计算的 scores 按阈值筛选 (per-type th dict 或全局 float)。
    与 gating_predict 输出格式一致的纯 Python 循环, 但不调 model。
    """
    preds_list, labels_list = [], []
    for s, sc in zip(samples, scores_per_sample):
        kept = []
        for i, c in enumerate(s['candidates']):
            t = c['type']
            if t not in TYPE2IDX:
                continue
            th = threshold.get(t, 0.5) if isinstance(threshold, dict) else threshold
            if i < len(sc) and float(sc[i]) >= th:
                kept.append({
                    'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                    'type': t, 'entity': c['entity'],
                })
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


# ====================================================================
# 5c. 策略 A_v2: Consensus V2 Hard-Rule
# ====================================================================
# 共识细分 (per-beam 特征)
#   共识强: best_beam_idx=0 (beam 1 也预测了) + vote >= 4  → 必留
#   共识弱: best_beam_idx >= 3 (仅 beam 4-5 预测)           → 加 bert_conf 阈值
#   共识中: 默认                                              → 必留
#
#   单独 LLM: vote >= 5  AND  max_beam_conf >= th_llm
#                          AND  (max-min) <= 0.30   ← conf 分散度约束
# 输入参数 (per-type):
#   bsolo_th[type]         单独 BERT bert_conf 阈值
#   lsolo_th[type]         单独 LLM max_beam_conf 阈值
#   lsolo_min_vote[type]   单独 LLM 最小票数 (默认 5)
#   lsolo_max_spread[type] 单独 LLM conf 分散度上限 (默认 0.30)
def _a_v2_accept(c, bsolo, lsolo, lsolo_min_vote, lsolo_max_spread):
    """单个候选是否被 A_v2 接受 (基于 per-beam 投票 + conf 分散度)。
    接收的是 per-type 阈值字典与 c 候选, 返回 True/False。
    """
    t = c['type']
    if c['bert_present'] and c['llm_present']:
        # === 共识 (BERT+LLM 都预测了) ===
        # 弱共识: 仅 beam 4-5 预测, BERT 把握也小 → 需 bert_conf ≥ 0.50
        # 中/强共识: 必留
        best_bi = c.get('best_beam_idx', 0)
        if best_bi >= 3:
            return c['bert_conf'] >= 0.50
        return True
    if c['bert_present'] and not c['llm_present']:
        # === 单独 BERT ===
        return c['bert_conf'] >= bsolo.get(t, 0.95)
    if c['llm_present'] and not c['bert_present']:
        # === 单独 LLM ===
        # 投票数 + conf 分散度
        vote = c.get('vote_count', 0)
        if vote < lsolo_min_vote.get(t, 5):
            return False
        max_bc = c.get('best_beam_conf', c.get('llm_conf', 0.0))
        min_bc = min(c.get('beam_confs', [max_bc])) if c.get('beam_confs') else max_bc
        spread = max_bc - min_bc
        if spread > lsolo_max_spread.get(t, 0.30):
            return False
        return max_bc >= lsolo.get(t, 0.60)
    return False


def a_v2_predict(samples, bsolo_th, lsolo_th,
                 lsolo_min_vote=None, lsolo_max_spread=None):
    """A_v2 预测: per-beam 投票 + conf 分散度 + 共识细分。
    bsolo_th / lsolo_th : {type: float}
    lsolo_min_vote      : {type: int}    (默认 5)
    lsolo_max_spread    : {type: float}  (默认 0.30)
    """
    if lsolo_min_vote is None:
        lsolo_min_vote = {t: 5 for t in ALL_TYPES}
    if lsolo_max_spread is None:
        lsolo_max_spread = {t: 0.30 for t in ALL_TYPES}
    preds_list, labels_list = [], []
    for s in samples:
        kept = []
        for c in s['candidates']:
            if _a_v2_accept(c, bsolo_th, lsolo_th, lsolo_min_vote, lsolo_max_spread):
                kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                             'type': c['type'], 'entity': c['entity']})
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


# ====================================================================
# 策略 A_v2 训练/调用入口
# ====================================================================
def run_a_v2(train_samples, test_samples):
    """A_v2: Consensus V2 Hard-Rule 融合。
    流程:
      [1] 全局搜索 bsolo / lsolo / lsolo_min_vote / lsolo_max_spread
      [2] per-type 微调 (范围受限, 防过拟合)
      [3] 输出 val / test F1
    """
    print("\n" + "=" * 70)
    print("【策略 A_v2】Consensus V2")
    print("=" * 70)
    print("共识细分 (bi≥3 加 bert_conf) + 单独 LLM 加 conf 分散度约束")
    print("-" * 70)

    # [1] 诊断: 共识中/强 vs 弱
    both_n = both_hit = 0
    both_weak_n = both_weak_hit = 0
    for s in train_samples:
        lset = s['label_set']
        for c in s['candidates']:
            k = (c['start_idx'], c['entity'], c['type'])
            if c['bert_present'] and c['llm_present']:
                both_n += 1
                if k in lset: both_hit += 1
                if c.get('best_beam_idx', 0) >= 3:
                    both_weak_n += 1
                    if k in lset: both_weak_hit += 1
    print(f"  [1] 共识诊断 (train):")
    print(f"      共识全部:    n={both_n:5d}  hit={both_hit:5d}  prec={both_hit/both_n:.4f}")
    print(f"      共识弱 (bi>=3): n={both_weak_n:5d}  hit={both_weak_hit:5d}  prec={both_weak_hit/max(1,both_weak_n):.4f}")

    # [2] 全局超参网格搜索
    print(f"\n  [2] 全局网格搜索 (bsolo, lsolo, min_vote, max_spread):")
    best_f1, best_cfg = 0.0, None
    for bsolo_try in [0.90, 0.95]:
        for lsolo_try in [0.60, 0.70, 0.80, 0.85, 0.90]:
            for mv_try in [4, 5]:
                for sp_try in [0.20, 0.30, 0.50, 1.00]:
                    bsolo = {t: bsolo_try for t in ALL_TYPES}
                    lsolo = {t: lsolo_try for t in ALL_TYPES}
                    mv = {t: mv_try for t in ALL_TYPES}
                    sp = {t: sp_try for t in ALL_TYPES}
                    preds, labels = a_v2_predict(train_samples, bsolo, lsolo, mv, sp)
                    _, _, f1, _, _, _ = evaluate(preds, labels)
                    if f1 > best_f1:
                        best_f1, best_cfg = f1, (bsolo_try, lsolo_try, mv_try, sp_try)
    bsolo, lsolo, mv, sp = best_cfg
    print(f"  -> 全局最佳: bsolo={bsolo}, lsolo={lsolo}, min_vote={mv}, max_spread={sp}, "
          f"F1={best_f1:.4f}")

    # [3] per-type 微调 (范围小, 防过拟合)
    print(f"\n  [3] per-type lsolo 微调:")
    bsolo_th = {t: bsolo for t in ALL_TYPES}
    lsolo_th = {t: lsolo for t in ALL_TYPES}
    mv_th = {t: mv for t in ALL_TYPES}
    sp_th = {t: sp for t in ALL_TYPES}
    cur_f1 = best_f1
    for t in ALL_TYPES:
        best_l, best_v = lsolo, cur_f1
        for l_try in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
            lsolo_th[t] = l_try
            preds, labels = a_v2_predict(train_samples, bsolo_th, lsolo_th, mv_th, sp_th)
            _, _, f1, _, _, _ = evaluate(preds, labels)
            if f1 > best_v:
                best_v, best_l = f1, l_try
        lsolo_th[t] = best_l
        cur_f1 = best_v
    print(f"  -> per-type lsolo: {lsolo_th}")
    print(f"     val F1 = {cur_f1:.4f}")

    print(f"\n  [4] per-type bsolo 微调:")
    cur_f1_x = cur_f1
    for t in ALL_TYPES:
        best_b, best_v = bsolo, cur_f1_x
        for b_try in [0.85, 0.90, 0.95, 0.99]:
            bsolo_th[t] = b_try
            preds, labels = a_v2_predict(train_samples, bsolo_th, lsolo_th, mv_th, sp_th)
            _, _, f1, _, _, _ = evaluate(preds, labels)
            if f1 > best_v:
                best_v, best_b = f1, b_try
        bsolo_th[t] = best_b
        cur_f1_x = best_v
    print(f"  -> per-type bsolo: {bsolo_th}")
    print(f"     val F1 = {cur_f1_x:.4f}")

    # [5] 输出最终结果
    val_preds, val_labels = a_v2_predict(train_samples, bsolo_th, lsolo_th, mv_th, sp_th)
    test_preds, test_labels = a_v2_predict(test_samples, bsolo_th, lsolo_th, mv_th, sp_th)
    _, _, val_f1, val_tp, val_pred_n, val_true_n = evaluate(val_preds, val_labels)
    _, _, test_f1, test_tp, test_pred_n, test_true_n = evaluate(test_preds, test_labels)
    print(f"\n  [5] 验证集最终结果:")
    print_metrics("A_v2_refined (val)", *evaluate(val_preds, val_labels)[:3],
                  tp=val_tp, tp_pred=val_pred_n, tp_true=val_true_n)
    print_metrics("A_v2_refined (test)", *evaluate(test_preds, test_labels)[:3],
                  tp=test_tp, tp_pred=test_pred_n, tp_true=test_true_n)

    stats = {
        'both_n': both_n, 'both_weak_n': both_weak_n,
        'both_prec': both_hit/both_n, 'both_weak_prec': both_weak_hit/max(1,both_weak_n),
    }
    return val_f1, test_f1, bsolo_th, lsolo_th, mv_th, sp_th, stats


# ====================================================================
# 5d. 策略 A_cal: Calibrated Weighted Fusion (校准 + per-type α 加权)
# ====================================================================
def _fit_isotonic_calibrators(samples):
    """在 samples 上拟合 BERT / LLM conf 的 isotonic 校准器。
    输入: samples (list of {'candidates': [...], 'label_set': set(...)})
    返回: (bert_cal, llm_cal, n_bert, n_llm, n_bert_pos, n_llm_pos)
    校准器: cal.predict([x]) → P(正确|x), 0 ≤ y ≤ 1
    防呆: 若样本全部同一类 (正/负 例率 = 100% 或 = 0%), sklearn IsotonicRegression
          会因为只看到单 class 而报 "y must be at least two classes",
          此时 cal.f_ 仍为 None, 上层 predict 走 fast-path 不校准, 不会崩。
    """
    bert_X, bert_y, llm_X, llm_y = [], [], [], []
    for s in samples:
        lset = s['label_set']
        for c in s['candidates']:
            t = c['type']
            k = (c['start_idx'], c['entity'], t)
            is_correct = 1 if k in lset else 0
            if c.get('bert_present'):
                bert_X.append(float(c['bert_conf']))
                bert_y.append(is_correct)
            if c.get('llm_present'):
                llm_X.append(float(c['llm_conf']))
                llm_y.append(is_correct)
    bert_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    llm_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    n_bert, n_llm = len(bert_y), len(llm_y)
    n_bert_pos = int(sum(bert_y)) if bert_y else 0
    n_llm_pos = int(sum(llm_y)) if llm_y else 0
    if n_bert > 0 and len(set(bert_y)) > 1:
        bert_cal.fit(np.array(bert_X, dtype=np.float64),
                     np.array(bert_y, dtype=np.float64))
    if n_llm > 0 and len(set(llm_y)) > 1:
        llm_cal.fit(np.array(llm_X, dtype=np.float64),
                    np.array(llm_y, dtype=np.float64))
    return bert_cal, llm_cal, n_bert, n_llm, n_bert_pos, n_llm_pos


def _calibrate_samples_inplace(samples, bert_cal, llm_cal):
    """就地修改 samples 中 cand['bert_conf'] / cand['llm_conf'] 为校准后的 P(正确)。
    仅对有 present 的项校准, 无 present 的 conf 保持 0.0。
    重要: 此函数会破坏原始 conf, 调用方必须先 deep copy。
    """
    if not samples:
        return
    for s in samples:
        for c in s['candidates']:
            if (c.get('bert_present') and bert_cal is not None
                    and getattr(bert_cal, 'f_', None) is not None
                    and c.get('bert_conf', 0.0) > 0):
                c['bert_conf'] = float(bert_cal.predict([float(c['bert_conf'])])[0])
            if (c.get('llm_present') and llm_cal is not None
                    and getattr(llm_cal, 'f_', None) is not None
                    and c.get('llm_conf', 0.0) > 0):
                c['llm_conf'] = float(llm_cal.predict([float(c['llm_conf'])])[0])


def _cal_fusion_predict(samples, type_alphas, threshold):
    """校准 + per-type α 加权融合 (无共识加成, 无因子)。
    公式: score = α(类型) * cal_bert + (1-α) * cal_llm;  keep if score >= th
    输入 samples 必须是已校准的 (cand['bert_conf'/'llm_conf'] 已是 P(正确))。
    type_alphas: dict{type: float} (per-type) 或 float (全局 α)。
    """
    preds_list, labels_list = [], []
    for s in samples:
        kept = []
        for c in s['candidates']:
            t = c['type']
            alpha = type_alphas.get(t, 0.5) if isinstance(type_alphas, dict) else float(type_alphas)
            score = alpha * float(c.get('bert_conf', 0.0)) + (1.0 - alpha) * float(c.get('llm_conf', 0.0))
            if score >= threshold:
                kept.append({
                    'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                    'type': t, 'entity': c['entity'],
                })
        preds_list.append(kept)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


def _cal_fusion_predict_metrics(samples, type_alphas, threshold):
    """_cal_fusion_predict 的 fast 版, 返回 (tps, pns, tns) np.array。
    用途: 网格搜 ~66 次 evaluate 加速 (与 static_fusion_predict_metrics 对齐)。"""
    n = len(samples)
    tps = np.zeros(n, dtype=np.int64)
    pns = np.zeros(n, dtype=np.int64)
    tns = np.empty(n, dtype=np.int64)
    for i, s in enumerate(samples):
        lset = s['label_set']
        n_hit = n_keep = 0
        for c in s['candidates']:
            t = c['type']
            alpha = type_alphas.get(t, 0.5) if isinstance(type_alphas, dict) else float(type_alphas)
            score = alpha * float(c.get('bert_conf', 0.0)) + (1.0 - alpha) * float(c.get('llm_conf', 0.0))
            if score >= threshold:
                n_keep += 1
                k = (c['start_idx'], c['entity'], t)
                if k in lset:
                    n_hit += 1
        tps[i] = n_hit
        pns[i] = n_keep
        tns[i] = len(s['label_entities'])
    return tps, pns, tns


def run_calibrated_weighted_fusion(train_samples, test_samples, label='A_cal'):
    """策略 A_cal: Calibrated Weighted Fusion (校准 + per-type α 加权)。
    Steps:
      1) 拟合 isotonic 校准器 (在 train_samples 上)
      2) deep copy train/test samples, 应用校准 (in-place)
      3) 全局网格搜 (α, th) → per-type α 微调
      4) 输出 val / test P/R/F1
    返回: (val_f1, test_f1, type_alphas, threshold, train_cal, test_cal, cal_info)
    """
    import copy as _copy
    print("\n" + "=" * 70)
    print(f"【策略 {label}】Calibrated Weighted Fusion (校准 + per-type α 加权)")
    print("=" * 70)
    print("设计思路:")
    print("  1) Isotonic Regression 校准 BERT/LLM conf → P(正确|conf)")
    print("     解决 LLM conf 偏高/集中导致的加权偏差")
    print("  2) per-type α 加权:  score = α*cal_bert + (1-α)*cal_llm")
    print("  3) 网格搜 (α ∈ [0,1], th ∈ [0.2,0.7]) → per-type α 微调")
    print("  4) 单一阈值 th (无共识加成, 无 bert/llm_only 因子)")
    print("-" * 70)

    # 1) 拟合校准器
    print("\n  [1] 拟合 Isotonic 校准器 (在 train_samples 上) ...")
    bert_cal, llm_cal, n_b, n_l, n_b_pos, n_l_pos = _fit_isotonic_calibrators(train_samples)
    bert_rate = n_b_pos / n_b if n_b > 0 else 0.0
    llm_rate = n_l_pos / n_l if n_l > 0 else 0.0
    print(f"      BERT: {n_b} 样本, 正例率 = {bert_rate:.3f}, "
          f"校准器 fitted={bert_cal.f_ is not None}")
    print(f"      LLM:  {n_l} 样本, 正例率 = {llm_rate:.3f}, "
          f"校准器 fitted={llm_cal.f_ is not None}")
    cal_info = {
        'n_bert': n_b, 'n_llm': n_l,
        'bert_pos_rate': bert_rate, 'llm_pos_rate': llm_rate,
        'bert_cal_fitted': bert_cal.f_ is not None,
        'llm_cal_fitted': llm_cal.f_ is not None,
    }

    # 2) deep copy + 校准
    print("\n  [2] 应用校准 (deep copy 避免污染原 samples) ...")
    train_cal = _copy.deepcopy(train_samples)
    test_cal = _copy.deepcopy(test_samples)
    _calibrate_samples_inplace(train_cal, bert_cal, llm_cal)
    _calibrate_samples_inplace(test_cal, bert_cal, llm_cal)

    # 3) 全局网格搜 (α, th) — 校准后 conf ∈ [0,1], 阈值范围同 v3
    print("\n  [3] 全局网格搜 (α ∈ [0, 1] 步长 0.1, th ∈ [0.2, 0.7] 步长 0.05) ...")
    best_f1, best_th, best_alpha = 0.0, 0.4, 0.5
    type_alphas = {t: 0.5 for t in ALL_TYPES}
    alpha_range = list(np.arange(0.0, 1.05, 0.1))
    th_range = list(np.arange(0.20, 0.70, 0.05))
    total = len(alpha_range) * len(th_range)
    cnt, t0 = 0, time.time()
    for alpha in alpha_range:
        for th in th_range:
            cnt += 1
            tps, pns, tns = _cal_fusion_predict_metrics(train_cal, alpha, th)
            _, _, f1 = _f1_from_arr(tps, pns, tns)
            if f1 > best_f1:
                best_f1, best_th, best_alpha = f1, th, alpha
                for t in ALL_TYPES:
                    type_alphas[t] = alpha
            if cnt % 20 == 0 or cnt == total:
                print(f"      [{cnt}/{total}] 耗时 {time.time()-t0:.1f}s, "
                      f"当前最优 F1={best_f1:.4f}")
    print(f"  -> 全局最佳: α={best_alpha:.2f}, th={best_th:.2f}, F1={best_f1:.4f}")

    # 4) per-type α 微调
    print(f"\n  [4] per-type α 微调 (th 固定 = {best_th:.2f}, α ∈ [0,1] 步长 0.05) ...")
    for etype in ALL_TYPES:
        best_f1_t, best_alpha_t = 0.0, type_alphas[etype]
        for alpha in np.arange(0.0, 1.05, 0.05):
            test_alphas = dict(type_alphas)
            test_alphas[etype] = float(alpha)
            tps, pns, tns = _cal_fusion_predict_metrics(train_cal, test_alphas, best_th)
            _, _, f1 = _f1_from_arr(tps, pns, tns)
            if f1 > best_f1_t:
                best_f1_t, best_alpha_t = f1, float(alpha)
        type_alphas[etype] = best_alpha_t
        print(f"      {etype}: α={best_alpha_t:.2f}, F1={best_f1_t:.4f}")

    # 5) 验证集最终
    print(f"\n  [5] 验证集最终结果 (α 与阈值见上):")
    val_preds, val_labels = _cal_fusion_predict(train_cal, type_alphas, best_th)
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels, tol=POSITION_TOLERANCE)
    print_metrics(f"{label} (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    # 6) 测试集
    test_preds, test_labels = _cal_fusion_predict(test_cal, type_alphas, best_th)
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels, tol=POSITION_TOLERANCE)
    print_metrics(f"{label} (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)

    # 7) 校准贡献消融 — 关掉校准, 看纯 per-type α 加权能拿多少 F1
    print(f"\n  [7] 校准贡献消融 (用未校准 conf 跑同样搜索, ΔF1 即为校准贡献):")
    best_unc_f1, best_unc_th, best_unc_alpha = 0.0, 0.4, 0.5
    unc_alphas = {t: 0.5 for t in ALL_TYPES}
    for alpha in alpha_range:
        for th in th_range:
            tps, pns, tns = _cal_fusion_predict_metrics(train_samples, alpha, th)
            _, _, f1 = _f1_from_arr(tps, pns, tns)
            if f1 > best_unc_f1:
                best_unc_f1, best_unc_th, best_unc_alpha = f1, th, alpha
                for t in ALL_TYPES:
                    unc_alphas[t] = alpha
    for etype in ALL_TYPES:
        best_f1_t, best_alpha_t = 0.0, unc_alphas[etype]
        for alpha in np.arange(0.0, 1.05, 0.05):
            test_alphas = dict(unc_alphas)
            test_alphas[etype] = float(alpha)
            tps, pns, tns = _cal_fusion_predict_metrics(
                train_samples, test_alphas, best_unc_th)
            _, _, f1 = _f1_from_arr(tps, pns, tns)
            if f1 > best_f1_t:
                best_f1_t, best_alpha_t = f1, float(alpha)
        unc_alphas[etype] = best_alpha_t
    unc_test_preds, _ = _cal_fusion_predict(test_samples, unc_alphas, best_unc_th)
    _, _, unc_test_f1, _, _, _ = evaluate(unc_test_preds, test_labels, tol=POSITION_TOLERANCE)
    cal_delta = test_f1 - unc_test_f1
    cal_info['unc_alphas'] = unc_alphas
    cal_info['unc_th'] = best_unc_th
    cal_info['unc_test_f1'] = unc_test_f1
    print(f"      校准后   test F1 = {test_f1:.4f}  (α per-type, th={best_th:.2f})")
    print(f"      未校准   test F1 = {unc_test_f1:.4f}  (α per-type, th={best_unc_th:.2f})")
    print(f"      校准贡献 ΔF1     = {cal_delta:+.4f}")

    return val_f1, test_f1, type_alphas, best_th, train_cal, test_cal, cal_info


def _predict_with_calibrators_reuse_alphas(target_samples, train_for_fit,
                                            type_alphas, threshold):
    """在 train_for_fit 上重 fit 校准器, 重新校准 target_samples, 用已搜好的
    type_alphas / threshold 直接出 preds。
    用途: 严格评估块中, A_cal 的 val 评估不能用 train_cal (in-sample),
          必须用门控网络未见的 va_samples (8:2 切, 800 条)。
    """
    import copy as _copy
    if train_for_fit is None or len(train_for_fit) == 0:
        return _cal_fusion_predict(target_samples, type_alphas, threshold)[0]
    bert_cal, llm_cal, _, _, _, _ = _fit_isotonic_calibrators(train_for_fit)
    tgt = _copy.deepcopy(target_samples)
    _calibrate_samples_inplace(tgt, bert_cal, llm_cal)
    preds, _ = _cal_fusion_predict(tgt, type_alphas, threshold)
    return preds


def run_gating_fusion(train_samples, test_samples, llm_lines, bert_lines, source='ner_models',
                      epochs=30, lr=2e-3, save_dir='saved_models_clean',
                      drop_beam_features=False):
    """drop_beam_features=True: 训练门控网络时不喂 5-beam 特有特征 (B_b1 消融用)
    网络结构分两套:
      - 5b 模式 (drop_beam=False): 2×hidden=64 + 3 头 (w_bert/w_llm/bonus)
        融合分 = w_bert*bert_conf + w_llm*llm_conf + bonus*both_pres*0.30
        损失: BCE(combined, label) + 0.3 * 文本级 ranking
      - b1 模式 (drop_beam=True):  1×hidden=32 + 1 头, 单 logit
        融合分 = sigmoid(logit)
        损失: BCE(logit, label) + 0.15 * MSE(sigmoid(logit), 0.5*llm+0.5*bert)
        (具体切换在 train_gating 内按 model_tag=='b1' 自动处理)
    """
    # 内部传 model_tag 给 train_gating, 让 5b / b1 各自存到独立文件名
    model_tag = 'b1' if drop_beam_features else '5b'
    print("\n" + "=" * 70)
    print("【策略 B】Gating Network Fusion (门控网络)")
    print("=" * 70)
    if drop_beam_features:
        print("模式: b1 (Beam-1 only) — 单头 32 维, sigmoid(logit) 直接出分")
        print("      损失: BCE + 0.15 * MSE(sigmoid(logit), 0.5*llm+0.5*bert)")
    else:
        print("模式: 5b — 3 头 (w_bert, w_llm, bonus)")
        print("      融合分 = w_bert*bert_conf + w_llm*llm_conf + bonus*both_pres*0.30")
        print("      损失: BCE(combined, label) + 0.3 * 文本级 ranking")
    print("-" * 70)

    print("  [1] 训练/验证切分 (8:2) ...")
    n = len(train_samples)
    rng = np.random.RandomState(GLOBAL_SEED)
    perm = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx = set(perm[:n_val].tolist())
    tr_samples = [s for i, s in enumerate(train_samples) if i not in val_idx]
    va_samples = [s for i, s in enumerate(train_samples) if i in val_idx]

    feats_tr, tids_tr, gids_tr, labs_tr = _candidates_to_features(tr_samples, drop_beam_features=drop_beam_features)
    feats_va, tids_va, gids_va, labs_va = _candidates_to_features(va_samples, drop_beam_features=drop_beam_features)
    print(f"      训练候选: {len(feats_tr)}  验证候选: {len(feats_va)}")
    train_ds = TensorDataset(*_to_tensors(feats_tr, tids_tr, gids_tr, labs_tr,
                                            drop_beam_features=drop_beam_features))
    valid_ds = TensorDataset(*_to_tensors(feats_va, tids_va, gids_va, labs_va,
                                            drop_beam_features=drop_beam_features))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False)

    print(f"\n  [2] 训练门控网络 (epochs={epochs}, lr={lr}, drop_beam={drop_beam_features}) ...")
    n_feats = 12 if drop_beam_features else 16
    model = GatingNetwork(num_types=NUM_TYPES, n_feats=n_feats)
    t0 = time.time()
    model = train_gating(model, train_loader, valid_loader,
                         epochs=epochs, lr=lr, patience=5, save_dir=save_dir,
                         model_tag=model_tag)
    print(f"      训练耗时: {time.time() - t0:.1f}s")

    print("\n  [3] 全局阈值网格搜索 (在验证集上, 含 P/R 平衡):")
    # 加速: 一次批量推理拿所有样本分数, 后续 th 扫描复用 (避免 ~40×800 次 model forward)
    val_scores = _gating_batch_scores(va_samples, model, drop=getattr(model, 'drop_beam_features', False))
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.05, 0.85, 0.02):
        preds_list, labels_list = [], []
        for s, sc in zip(va_samples, val_scores):
            kept = []
            for i, c in enumerate(s['candidates']):
                t = c['type']
                if t not in TYPE2IDX:
                    continue
                if i < len(sc) and float(sc[i]) >= float(th):
                    kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                 'type': t, 'entity': c['entity']})
            preds_list.append(kept)
            labels_list.append(s['label_entities'])
        p, r, f1, _, _, _ = evaluate(preds_list, labels_list)
        marker = " *" if f1 > best_f1 else ""
        if f1 >= best_f1 - 0.005:
            print(f"      th={th:.2f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}{marker}")
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    # 同时找出 P≈R 的阈值 (差距最小), 备选 — 复用 val_scores, 不再重算
    best_balanced_th, best_balanced_gap = 0.5, 1.0
    best_balanced_f1 = best_f1
    for th in np.arange(0.05, 0.85, 0.02):
        preds_list, labels_list = [], []
        for s, sc in zip(va_samples, val_scores):
            kept = []
            for i, c in enumerate(s['candidates']):
                t = c['type']
                if t not in TYPE2IDX:
                    continue
                if i < len(sc) and float(sc[i]) >= float(th):
                    kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                 'type': t, 'entity': c['entity']})
            preds_list.append(kept)
            labels_list.append(s['label_entities'])
        p, r, f1, _, _, _ = evaluate(preds_list, labels_list)
        gap = abs(p - r)
        if gap < best_balanced_gap:
            best_balanced_gap, best_balanced_th, best_balanced_f1 = gap, float(th), f1
    print(f"  -> 全局 F1 最佳: th={best_th:.2f}, F1={best_f1:.4f}")
    print(f"  -> P/R 最平衡: th={best_balanced_th:.2f}  (P-R 差={best_balanced_gap:.4f})")
    # 如果平衡阈值 F1 不显著差, 用平衡阈值
    if best_balanced_f1 >= best_f1 - 0.003:
        print(f"  -> 采用 P/R 平衡阈值 {best_balanced_th} (F1={best_balanced_f1:.4f}, 平衡更优)")
        best_th = best_balanced_th

    # 注: gating 网络的 combined score 已经是 per-sample 的连续可分值, 全局阈值即可。
    print("\n  [4] per-type 微调 (范围 -0.10 ~ +0.10, 防止过拟合):")
    best_type_th = {t: best_th for t in ALL_TYPES}
    OFFSET = 0.10
    for etype in ALL_TYPES:
        best_t, best_f_t = best_th, 0.0
        for th in np.arange(best_th - OFFSET, best_th + OFFSET + 1e-9, 0.01):
            test_th = dict(best_type_th)
            test_th[etype] = float(th)
            preds_list, labels_list = [], []
            for s, sc in zip(va_samples, val_scores):
                kept = []
                for i, c in enumerate(s['candidates']):
                    t = c['type']
                    if t not in TYPE2IDX:
                        continue
                    th_t = test_th.get(t, 0.5)
                    if i < len(sc) and float(sc[i]) >= th_t:
                        kept.append({'start_idx': c['start_idx'], 'end_idx': c['end_idx'],
                                     'type': t, 'entity': c['entity']})
                preds_list.append(kept)
                labels_list.append(s['label_entities'])
            _, _, f1, _, _, _ = evaluate(preds_list, labels_list)
            if f1 > best_f_t:
                best_f_t, best_t = f1, float(th)
        best_type_th[etype] = best_t
        print(f"      {etype}: th={best_t:.2f}, F1={best_f_t:.4f}")

    # 验证集 / 测试集最终结果: 复用 val_scores, 重新算 test_scores (一次性)
    print("\n  [5] 验证集最终结果:")
    val_preds, val_labels = _filter_by_threshold(va_samples, val_scores, best_type_th)
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels)
    print_metrics("Gating-Network (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    test_scores = _gating_batch_scores(test_samples, model, drop=getattr(model, 'drop_beam_features', False))
    test_preds, test_labels = _filter_by_threshold(test_samples, test_scores, best_type_th)
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels)
    print_metrics("Gating-Network (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)
    # 返回 (val_f1, test_f1, type_th, (val_p/r, test_p/r), model) — model 用于论文严格评估重算
    return val_f1, test_f1, best_type_th, (val_p, val_r, test_p, test_r), model, va_samples


# ====================================================================
# 7. 主流程
# ====================================================================
def main():
    # ---------- 配置 ----------
    # 在这里选择数据集 (与原文件 main 中可选项对应)
    LLM_FILE = "evl_f1/mark/glm4_ner_confidence_beams5.jsonl"
    #LLM_FILE = "evl_f1/mark/lla_ner_Llama3-8B-confidence_beams_5.jsonl"
    BERT_FILE = "evl_f1/EGP_chinese-roberta-wwm-ext.jsonl"
    BERT_FILE = "evl_f1/TP_chinese-roberta-wwm-ext.jsonl"

    LLM_SOURCE = 'ner_models'
    USE_BEAM = True  # 5-beam 投票

    TRAIN_SAMPLE_SIZE = 4000  # 训练 (含验证) 样本数
    TEST_SAMPLE_SIZE = 1000   # 测试样本数
    EPOCHS = 30               # 门控网络训练轮数
    LR = 2e-3
    SAVE_DIR = "saved_models_clean"

    print("=" * 70)
    print("LLM + BERT 实体融合 (重写整洁版)")
    print("=" * 70)
    print(f"  LLM 文件:  {LLM_FILE}")
    print(f"  BERT 文件: {BERT_FILE}")
    print(f"  LLM 数据来源: {LLM_SOURCE}")
    print(f"  Beam 投票:    {USE_BEAM}")
    print(f"  位置容差:    {POSITION_TOLERANCE}")
    print()

    # 1) 加载 + 切分
    print("[Step 1] 加载数据并切分训练/测试集 ...")
    llm_lines, bert_lines = load_data(LLM_FILE, BERT_FILE)
    print(f"  加载完成: {len(llm_lines)} 条 (以较少的为准)")
    (train_llm, train_bert), (test_llm, test_bert) = split_train_test(
        llm_lines, bert_lines, test_size=TEST_SAMPLE_SIZE, seed=GLOBAL_SEED)
    # 控制训练样本数
    train_llm = train_llm[:TRAIN_SAMPLE_SIZE]
    train_bert = train_bert[:TRAIN_SAMPLE_SIZE]
    print(f"  训练池 (供门控网络 8:2 切): {len(train_llm)} 条")
    print(f"  测试:                       {len(test_llm)} 条")
    # 注: 验证集 (val) 在 run_gating_fusion 内部 8:2 切出 (n=tr//5), 不在 main 切

    # 2) 解析 → samples (先用 α=0.7, β=0.05 占位, 后面 [Step 1.5] 搜最优后重 build)
    print("\n[Step 2] 解析训练集 (初次, α=0.7 β=0.05 占位) ...")
    global LLM_ALPHA, LLM_VOTE_REWARD
    LLM_ALPHA = 0.7
    LLM_VOTE_REWARD = 0.05
    train_samples = build_samples(train_llm, train_bert, source=LLM_SOURCE,
                                  tol=POSITION_TOLERANCE, use_beam=USE_BEAM)
    print(f"  训练样本 (有效): {len(train_samples)}")
    print("\n[Step 3] 解析测试集 (初次, α=0.7 β=0.05 占位) ...")
    test_samples = build_samples(test_llm, test_bert, source=LLM_SOURCE,
                                 tol=POSITION_TOLERANCE, use_beam=USE_BEAM)
    print(f"  测试样本 (有效): {len(test_samples)}")

    # 3.1) 【Beam-1 only 副本】用于消融实验: 模拟"LLM 服务只返回 1 个" 的公平对照
    #     parse_llm_line 的 use_beam=False 分支已改为 _parse_mark_format_beam(models[:1], ...),
    #     所以 cand['llm_conf'] = Beam-1 真实 conf, vote_count=1, 无 5-beam 投票信号
    print("\n[Step 3.1] 解析 Beam-1 only 副本 (用于消融: A_b1 / B_b1) ...")
    train_samples_b1 = build_samples(train_llm, train_bert, source=LLM_SOURCE,
                                      tol=POSITION_TOLERANCE, use_beam=False)
    test_samples_b1 = build_samples(test_llm, test_bert, source=LLM_SOURCE,
                                     tol=POSITION_TOLERANCE, use_beam=False)
    print(f"  Beam-1 训练样本: {len(train_samples_b1)}  测试样本: {len(test_samples_b1)}")

    # 1.5) 搜索最优 (α, β) (max+mean+vote_reward), 用 A 策略简化版的 train F1 选
    # A 策略对 llm_conf 变化敏感, 比 A'' 适合做聚合评估
    # 注: 此步搜出的 (α, β) 用在 train_samples (3000) 上, 严格评估块的 val 列在门控网络未见的 800 条上
    if USE_BEAM:
        print("\n[Step 1.5] 搜索 (α, β) 参数 (max+mean+vote_reward), 以 A 策略简化版在 train_samples 上的 F1 为目标 ...")
        # 加速: 30 次网格只重算 conf (调 _reaggregate_llm_confs), 不重 build_samples
        import copy
        train_samples_base = copy.deepcopy(train_samples)  # 原始 llm_conf 作模板
        # 用 A 策略简化版 (α_per_type=F1_bert/(F1_bert+F1_llm), bonus=0.05, th=0.5)
        def _quick_a_f1(samples):
            type_alphas = {}
            for tt in ALL_TYPES:
                ft_b = _type_only_f1(samples, tt, source='bert')
                ft_l = _type_only_f1(samples, tt, source='llm')
                type_alphas[tt] = (ft_b / max(1e-6, ft_b + ft_l)) if (ft_b + ft_l) > 0 else 0.5
            type_thresholds = {tt: 0.5 for tt in ALL_TYPES}
            preds, labels = static_fusion_predict(
                samples, type_alphas, type_thresholds,
                consensus_bonus=0.05, consensus_th_mult=0.9,
                bert_only_factor=0.9, llm_only_factor=0.7)
            _, _, f1, _, _, _ = evaluate(preds, labels, tol=POSITION_TOLERANCE)
            return f1

        best_alpha, best_beta, best_alphabeta_f1 = 0.7, 0.05, 0.0
        # β 搜索扩到 [-0.05, +0.10] 含 0, 用来确认"vote_reward 软奖励在 A 策略上确实无信号"
        for alpha_try in [1.00, 0.85, 0.70, 0.55, 0.40]:
            for beta_try in [-0.05, 0.00, 0.03, 0.05, 0.08, 0.10]:
                # 每次从 base 拷贝, 重算 conf (in-place 但 base 不变)
                _train = copy.deepcopy(train_samples_base)
                _reaggregate_llm_confs(_train, alpha_try, beta_try)
                _f1 = _quick_a_f1(_train)
                marker = " *" if _f1 > best_alphabeta_f1 else ""
                print(f"      α={alpha_try:.2f}  β={beta_try:.2f}  A val F1={_f1:.4f}{marker}")
                if _f1 > best_alphabeta_f1:
                    best_alphabeta_f1, best_alpha, best_beta = _f1, alpha_try, beta_try
        print(f"  -> 最优 α={best_alpha:.2f}, β={best_beta:.2f}  (A val F1 = {best_alphabeta_f1:.4f})")
        LLM_ALPHA = best_alpha
        LLM_VOTE_REWARD = best_beta
        # 用最优参数重算 train/test 的 llm_conf (不再重 build_samples)
        print(f"\n  [重算 conf] 用 α={best_alpha}, β={best_beta} 重新聚合 LLM conf ...")
        _reaggregate_llm_confs(train_samples, best_alpha, best_beta)
        _reaggregate_llm_confs(test_samples,  best_alpha, best_beta)

    # 2b) 打印 beam 投票分布
    if USE_BEAM:
        from collections import Counter
        vote_dist = Counter()
        for s in train_samples + test_samples:
            for c in s['candidates']:
                if c['llm_present']:
                    vote_dist[int(c.get('vote_count', 1))] += 1
        print("\n  [Beam 投票分布] (LLM-only 候选的 beam 票数):")
        for v in sorted(vote_dist):
            print(f"    {v} 票: {vote_dist[v]} 候选")
        # 共识 (BERT+LLM 共同) 的平均票数
        both_votes = [c.get('vote_count', 1) for s in train_samples + test_samples
                      for c in s['candidates'] if c['bert_present'] and c['llm_present']]
        if both_votes:
            print(f"    共识候选的平均票数: {sum(both_votes) / len(both_votes):.2f} "
                  f"(共 {len(both_votes)} 个)")
        solo_votes = [c.get('vote_count', 1) for s in train_samples + test_samples
                      for c in s['candidates']
                      if c['llm_present'] and not c['bert_present']]
        if solo_votes:
            print(f"    单独 LLM 候选的平均票数: {sum(solo_votes) / len(solo_votes):.2f} "
                  f"(共 {len(solo_votes)} 个)")

    # 3) Baseline 指标
    print("\n" + "=" * 70)
    print("【基线】单模型")
    print("=" * 70)
    bert_preds, bert_labels = bert_only_predict(train_samples, train_llm, train_bert, source=LLM_SOURCE)
    bp_v, br_v, bf1_v, btp_v, bpn_v, btn_v = evaluate(bert_preds, bert_labels)
    print_metrics("BERT Only (val)", bp_v, br_v, bf1_v, btp_v, bpn_v, btn_v)
    bert_test_preds, bert_test_labels = bert_only_predict(test_samples, test_llm, test_bert, source=LLM_SOURCE)
    bp, br, bf1_t, btp, bpn, btn = evaluate(bert_test_preds, bert_test_labels)
    print_metrics("BERT Only (test)", bp, br, bf1_t, btp, bpn, btn)

    # 3) LLM Only 多个基线 (Top-1 ~ Top-5 + 5-union, 论文严格基线是 Top-1)
    print("\n" + "=" * 90)
    print("【LLM Only 基线 · 严格拆分】 (LLM 服务对外只返回 1 个最佳 → 论文主基线 = Top-1)")
    print("=" * 90)
    print(f"  {'基线':<24} | {'val (train)':^30} | {'test':^30}")
    print(f"  {'':<24} | {'P':>7} {'R':>7} {'F1':>7} | {'P':>7} {'R':>7} {'F1':>7}")
    print("  " + "-" * 78)
    # Top-1 (论文主基线) ~ Top-5
    llm_baselines = {}
    for k in range(5):
        v_preds, v_labels = llm_only_predict(train_samples, beam_idx=k)
        p, r, f1, _, _, _ = evaluate(v_preds, v_labels)
        t_preds, t_labels = llm_only_predict(test_samples, beam_idx=k)
        tp, tr, tf1, _, _, _ = evaluate(t_preds, t_labels)
        llm_baselines[k] = (p, r, f1, tp, tr, tf1)
        print(f"  LLM Top-{k+1} (Beam {k+1})".ljust(26) +
              f" | {p:>7.4f} {r:>7.4f} {f1:>7.4f} | {tp:>7.4f} {tr:>7.4f} {tf1:>7.4f}")
    # 5-union 不加阈值 (原行为, 不严谨)
    v_preds, v_labels = llm_only_predict(train_samples, beam_idx=None)
    p, r, f1, _, _, _ = evaluate(v_preds, v_labels)
    t_preds, t_labels = llm_only_predict(test_samples, beam_idx=None)
    tp, tr, tf1, _, _, _ = evaluate(t_preds, t_labels)
    print(f"  LLM 5-Union (no-thr)  ".ljust(26) +
          f" | {p:>7.4f} {r:>7.4f} {f1:>7.4f} | {tp:>7.4f} {tr:>7.4f} {tf1:>7.4f}"
          f"  ← 当前原 'LLM Only' 基线")
    llm_baselines['union'] = (p, r, f1, tp, tr, tf1)
    # 5-union + conf 阈值 (Top-1 风格的严格做法)
    v_preds, v_labels = llm_only_predict(train_samples, beam_idx=None, conf_th=0.5)
    p, r, f1, _, _, _ = evaluate(v_preds, v_labels)
    t_preds, t_labels = llm_only_predict(test_samples, beam_idx=None, conf_th=0.5)
    tp, tr, tf1, _, _, _ = evaluate(t_preds, t_labels)
    print(f"  LLM 5-Union (conf≥0.5)".ljust(26) +
          f" | {p:>7.4f} {r:>7.4f} {f1:>7.4f} | {tp:>7.4f} {tr:>7.4f} {tf1:>7.4f}"
          f"  ← conf 过滤后")
    print("  " + "-" * 78)
    # 论文主基线: Top-1
    lp_v, lr_v, lf1_v, lp, lr, lf1_t = llm_baselines[0]

    # 4) 策略 A: 静态权重融合 (3-case 软融合)
    a_val, a_test, a_alphas, a_th, a_hyper = run_static_fusion(train_samples, test_samples)
    # 重算 val/test P/R (汇总用, 必须用 a_hyper 里的网格搜出超参, 不能 hardcoded 否则会与 run 内部不一致)
    a_v_preds, _ = static_fusion_predict(
        train_samples, a_alphas, a_th,
        consensus_bonus=a_hyper['bonus'], consensus_th_mult=a_hyper['mult'],
        bert_only_factor=a_hyper['bf'], llm_only_factor=a_hyper['lf'],
        vote_bonus_coef=a_hyper['vbc'])
    a_val_p, a_val_r, _, _, _, _ = evaluate(a_v_preds,
        [s['label_entities'] for s in train_samples])
    a_t_preds, _ = static_fusion_predict(
        test_samples, a_alphas, a_th,
        consensus_bonus=a_hyper['bonus'], consensus_th_mult=a_hyper['mult'],
        bert_only_factor=a_hyper['bf'], llm_only_factor=a_hyper['lf'],
        vote_bonus_coef=a_hyper['vbc'])
    a_test_p, a_test_r, _, _, _, _ = evaluate(a_t_preds,
        [s['label_entities'] for s in test_samples])

    # 4b) 策略 A'': Vote-Aware Hard-Rule
    app_val, app_test, app_th, app_mvb, app_mvl = run_static_fusion_voteaware(
        train_samples, test_samples)
    app_vp, app_vr, _, _, _, _ = evaluate(
        static_fusion_voteaware_predict(train_samples, app_th, app_mvb, app_mvl)[0],
        [s['label_entities'] for s in train_samples])
    app_tp, app_tr, _, _, _, _ = evaluate(
        static_fusion_voteaware_predict(test_samples, app_th, app_mvb, app_mvl)[0],
        [s['label_entities'] for s in test_samples])

    # 4c) 策略 A_v2: Consensus V2
    av2_val, av2_test, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d, av2_stats = run_a_v2(
        train_samples, test_samples)
    av2_vp, av2_vr, _, _, _, _ = evaluate(
        a_v2_predict(train_samples, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d)[0],
        [s['label_entities'] for s in train_samples])
    av2_tp, av2_tr, _, _, _, _ = evaluate(
        a_v2_predict(test_samples, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d)[0],
        [s['label_entities'] for s in test_samples])

    # 4d) 策略 A_cal: Calibrated Weighted Fusion (校准 + per-type α 加权)
    #     来自 gating_network_dp_mark_v3.py 策略3/9, 用 train_samples 拟合校准器,
    #     deep copy 后的 train_cal / test_cal 走纯 2-特征加权 (无共识加成)
    #     注: 5-beam 模式下 c['llm_conf'] 已是 α·max+(1-α)·mean+β·(N-1) 聚合,
    #         校准仍能学习"高 conf 是否真" → 减少 LLM 主导偏差
    acal_val, acal_test, acal_alphas, acal_th, train_cal, test_cal, acal_info = \
        run_calibrated_weighted_fusion(train_samples, test_samples, label='A_cal')
    # 重算 val/test P/R (汇总用, 必须用 acal_alphas / acal_th 内部搜出的超参)
    acal_v_preds, _ = _cal_fusion_predict(train_cal, acal_alphas, acal_th)
    acal_val_p, acal_val_r, _, _, _, _ = evaluate(
        acal_v_preds, [s['label_entities'] for s in train_cal], tol=POSITION_TOLERANCE)
    acal_t_preds, _ = _cal_fusion_predict(test_cal, acal_alphas, acal_th)
    acal_test_p, acal_test_r, _, _, _, _ = evaluate(
        acal_t_preds, [s['label_entities'] for s in test_cal], tol=POSITION_TOLERANCE)

    # 5) 策略 B: 门控网络融合
    b_val, b_test, b_th, (b_vp, b_vr, b_tp, b_tr), gating_model, va_samples = run_gating_fusion(
        train_samples, test_samples,
        train_llm, train_bert,
        source=LLM_SOURCE, epochs=EPOCHS, lr=LR, save_dir=SAVE_DIR,
    )

    # 5.5) LLM 5 个 beam 各自的 P/R/F1 (供对比)
    print("\n" + "=" * 90)
    print("【LLM 5-Beam 单独性能】 (按 beam 索引, 0=beam1 准确度最高)")
    print("=" * 90)
    beam_results = []
    for k in range(5):
        bv_p, bv_r, bv_f1, _, _, _ = eval_llm_beam_k(train_llm, train_bert, beam_idx=k, source=LLM_SOURCE)
        bt_p, bt_r, bt_f1, _, _, _ = eval_llm_beam_k(test_llm,  test_bert,  beam_idx=k, source=LLM_SOURCE)
        beam_results.append((bv_p, bv_r, bv_f1, bt_p, bt_r, bt_f1))
        print(f"  LLM Beam-{k+1} (val)  P={bv_p:.4f}  R={bv_r:.4f}  F1={bv_f1:.4f}")
        print(f"  LLM Beam-{k+1} (test) P={bt_p:.4f}  R={bt_r:.4f}  F1={bt_f1:.4f}")

    # 5.6) 5-beam 投票 (5 取并集) vs 仅单 beam 的对比
    print("\n【5-beam 取并集 (无阈值过滤)】")
    union_preds_v, union_labels_v = [], []
    union_preds_t, union_labels_t = [], []

    def _union_pred_and_llm_label(ll, bl):
        """5-beam union preds + LLM 文件 label 字段 (mark 格式, 人工标注)"""
        ll_obj = _safe_parse_json(ll)
        if ll_obj is None:
            return [], []
        text = ll_obj.get('text', '')
        models = ll_obj.get(LLM_SOURCE, []) or ll_obj.get('ner_models', [])
        union_ents = {}
        for m in models:
            for k2, v2 in _parse_mark_format_predict(
                    m.get('predict', ''), text, default_conf=0.8).items():
                union_ents[k2] = v2
        preds = [
            {'start_idx': e['start_idx'], 'end_idx': e['end_idx'],
             'type': e['type'], 'entity': e['entity']} for e in union_ents.values()
        ]
        # ground truth 用 LLM 文件 label 字段 (人工标注), 不是 BERT 文件 (BERT 是推理结果)
        label_text = ll_obj.get('label', '')
        raw_ents = parse_marked_text_with_pos(label_text)
        raw_ents = _correct_entity_positions(raw_ents, text)
        labels = [
            {'start_idx': s, 'end_idx': e, 'entity': txt, 'type': t}
            for s, e, txt, t, _ in raw_ents
        ]
        return preds, labels

    for ll, bl in zip(train_llm, train_bert):
        preds, labels = _union_pred_and_llm_label(ll, bl)
        union_preds_v.append(preds)
        union_labels_v.append(labels)
    for ll, bl in zip(test_llm, test_bert):
        preds, labels = _union_pred_and_llm_label(ll, bl)
        union_preds_t.append(preds)
        union_labels_t.append(labels)
    uvp, uvr, uvf1, _, _, _ = evaluate(union_preds_v, union_labels_v)
    utp, utr, utf1, _, _, _ = evaluate(union_preds_t, union_labels_t)
    print(f"  5-beam union (val)  P={uvp:.4f}  R={uvr:.4f}  F1={uvf1:.4f}")
    print(f"  5-beam union (test) P={utp:.4f}  R={utr:.4f}  F1={utf1:.4f}")

    # ============================================================
    # 5.7) 【消融 A_b1 / B_b1】用 Beam-1 only 构造 samples
    #     - cand['llm_conf'] = Beam-1 真实 conf (无 5-beam 聚合)
    #     - cand['vote_count'] = 1 (无多 beam 投票信号)
    #     目的: 对照"5 beam 投票"与"单 beam"的差异, 量化投票对融合的贡献
    # ============================================================
    print("\n" + "=" * 90)
    print("【消融 A_b1 / B_b1】用 Beam-1 only (cand.llm_conf = Beam-1 真实 conf, vote_count=1)")
    print("=" * 90)

    # 5.7a) A_b1: 静态权重融合 (Beam-1 only)
    print("\n  [A_b1] 静态权重融合 · Beam-1 only")
    ab1_val, ab1_test, ab1_alphas, ab1_th, ab1_hyper = run_static_fusion(
        train_samples_b1, test_samples_b1)
    ab1_v_preds, _ = static_fusion_predict(
        train_samples_b1, ab1_alphas, ab1_th,
        consensus_bonus=ab1_hyper['bonus'], consensus_th_mult=ab1_hyper['mult'],
        bert_only_factor=ab1_hyper['bf'], llm_only_factor=ab1_hyper['lf'],
        vote_bonus_coef=ab1_hyper['vbc'])
    ab1_vp, ab1_vr, ab1_vf1, _, _, _ = evaluate(
        ab1_v_preds, [s['label_entities'] for s in train_samples_b1])
    ab1_t_preds, _ = static_fusion_predict(
        test_samples_b1, ab1_alphas, ab1_th,
        consensus_bonus=ab1_hyper['bonus'], consensus_th_mult=ab1_hyper['mult'],
        bert_only_factor=ab1_hyper['bf'], llm_only_factor=ab1_hyper['lf'],
        vote_bonus_coef=ab1_hyper['vbc'])
    ab1_tp, ab1_tr, ab1_tf1, _, _, _ = evaluate(
        ab1_t_preds, [s['label_entities'] for s in test_samples_b1])
    print(f"  A_b1 (val=train)  P={ab1_vp:.4f}  R={ab1_vr:.4f}  F1={ab1_vf1:.4f}")
    print(f"  A_b1 (test)       P={ab1_tp:.4f}  R={ab1_tr:.4f}  F1={ab1_tf1:.4f}")

    # 5.7b) B_b1: 门控网络 (Beam-1 only)
    # 关键: Beam-1 模式下 cand['vote_count']=1, cand['llm_max_conf']=cand['llm_avg_conf']=cand['llm_conf'],
    #       cand['best_beam_idx']=0 — 这 4 个"5 beam 特有"特征全退化为常数, 直接喂会让模型学到"vote=1=不可信"而崩。
    # 修复: 用 drop_beam_features=True 让门控网络不喂这 4 个特征, 强制只用通用 12 维特征。
    # 又: 为对齐 gating_network_dp_mark_v3.py 的 Gating Network 效果, 此处使用更稳的
    #     超参 (epochs=100, lr=5e-4, patience=15) + BCE+0.15*MSE 蒸馏损失
    #     (在 train_gating 内部按 model_tag=='b1' 自动切换)
    print("\n  [B_b1] Gating Network · Beam-1 only (drop_beam_features=True)")
    bb1_val, bb1_test, bb1_th, (bb1_vp, bb1_vr, bb1_tp, bb1_tr), bb1_model, bb1_va = run_gating_fusion(
        train_samples_b1, test_samples_b1,
        train_llm, train_bert,
        source=LLM_SOURCE, epochs=100, lr=5e-4, save_dir=SAVE_DIR,
        drop_beam_features=True,
    )
    # bb1_va 已是 b1 样本的 8:2 切分验证集 (run_gating_fusion 内部用 GLOBAL_SEED 切的)
    # A_b1 严格评估也用同一份, 保证两个 b1 策略的 val/test 来自同一划分
    va_samples_b1 = bb1_va

    # 5.7c) A_cal_b1: Calibrated Weighted Fusion (Beam-1 only)
    #     用 train_samples_b1 重 fit 校准器 — 不能直接用 5-beam 的校准器,
    #     因 LLM conf 分布 (单 beam vs 5-beam 聚合) 不同 → 校准尺度不一致
    #     该消融用于回答"v3 log +0.0033 在 Beam-1 模式下能否在 st8 复现"
    print("\n  [A_cal_b1] Calibrated Weighted Fusion · Beam-1 only")
    acalb1_val, acalb1_test, acalb1_alphas, acalb1_th, train_cal_b1, test_cal_b1, acalb1_info = \
        run_calibrated_weighted_fusion(train_samples_b1, test_samples_b1, label='A_cal_b1')
    acalb1_v_preds, _ = _cal_fusion_predict(train_cal_b1, acalb1_alphas, acalb1_th)
    acalb1_vp, acalb1_vr, acalb1_vf1, _, _, _ = evaluate(
        acalb1_v_preds, [s['label_entities'] for s in train_cal_b1], tol=POSITION_TOLERANCE)
    acalb1_t_preds, _ = _cal_fusion_predict(test_cal_b1, acalb1_alphas, acalb1_th)
    acalb1_tp, acalb1_tr, acalb1_tf1, _, _, _ = evaluate(
        acalb1_t_preds, [s['label_entities'] for s in test_cal_b1], tol=POSITION_TOLERANCE)
    print(f"  A_cal_b1 (val=train)  P={acalb1_vp:.4f}  R={acalb1_vr:.4f}  F1={acalb1_vf1:.4f}")
    print(f"  A_cal_b1 (test)       P={acalb1_tp:.4f}  R={acalb1_tr:.4f}  F1={acalb1_tf1:.4f}")
    # 严格评估块的 b1 验证集也用同一份 (8:2 切, 与 B_b1 一致)

    # 6) 汇总 (P / R / F1 全列)
    # 注: 此表 val 列 = train_samples (3000) 上的 F1, 包含门控网络训练样本 (有数据泄露)
    # 真未见 val 性能见后文 "【论文严格评估】" 块 (val=va_samples, 800 条, 门控网络未见的)
    print("\n" + "=" * 90)
    print("最终结果汇总 (P / R / F1) — val 列=train_samples(3000), 真未见 val 见严格评估块")
    print("=" * 90)
    hdr = (f"  {'策略':<36} | {'val (=train)':^20} | {'test':^20} | {'Δ test':>8}")
    print(hdr)
    sub = (f"  {'':<36} | {'P':>6} {'R':>6} {'F1':>6} | "
           f"{'P':>6} {'R':>6} {'F1':>6} | {'F1':>8}")
    print(sub)
    print("  " + "-" * 86)
    def row(name, vp, vr, vf, tp, tr, tf):
        delta = tf - bf1_t
        return (f"  {name:<36} | {vp:>6.4f} {vr:>6.4f} {vf:>6.4f} | "
                f"{tp:>6.4f} {tr:>6.4f} {tf:>6.4f} | {delta:>+8.4f}")
    print(row("[基线] BERT Only", bp_v, br_v, bf1_v, bp, br, bf1_t))
    print(row("[基线] LLM Only",  lp_v, lr_v, lf1_v, lp, lr, lf1_t))
    # 5 个 LLM beam 各自基线 (按用户说明: beam 1 准确度最高, 5 最低)
    for k in range(5):
        bvp, bvr, bvf, btp, btr, btf = beam_results[k]
        print(row(f"[基线] LLM Beam-{k+1}", bvp, bvr, bvf, btp, btr, btf))
    print(row("[基线] LLM 5-beam union (无阈值)", uvp, uvr, uvf1, utp, utr, utf1))
    print(row("A. Static Weight", a_val_p, a_val_r, a_val, a_test_p, a_test_r, a_test))
    print(row("A''. Vote-Aware Hard-Rule",     app_vp, app_vr, app_val, app_tp, app_tr, app_test))
    print(row("A_v2. Consensus",               av2_vp, av2_vr, av2_val, av2_tp, av2_tr, av2_test))
    print(row("A_cal. Calibrated Weighted",    acal_val_p, acal_val_r, acal_val,
              acal_test_p, acal_test_r, acal_test))
    print(f"  [消融] A_cal 校准贡献: 校准后={acal_test:.4f} vs 未校准={acal_info.get('unc_test_f1', 0):.4f}"
          f" → Δ={acal_test - acal_info.get('unc_test_f1', 0):+.4f}")
    print(row("B. Gating Network",             b_vp, b_vr, b_val, b_tp, b_tr, b_test))
    print("  --- 消融: Beam-1 only (cand.llm_conf = Beam-1 真实 conf, 无 5 beam 投票) ---")
    # 注: 此处 val 列对 b1 策略是 in-sample (val=train_samples_b1, 含训练样本, 数据泄露),
    #     真未见 val (bb1_va) 性能见后文"【论文严格评估】"块的 A_b1/B_b1 行 — 与本表同名以便对比
    print(row("A_b1. Static (Beam-1)",     ab1_vp, ab1_vr, ab1_vf1, ab1_tp, ab1_tr, ab1_tf1))
    print(row("A_cal_b1. Calibrated (Beam-1)", acalb1_vp, acalb1_vr, acalb1_vf1,
              acalb1_tp, acalb1_tr, acalb1_tf1))
    print(f"  [消融] A_cal_b1 校准贡献: 校准后={acalb1_tf1:.4f} vs "
          f"未校准={acalb1_info.get('unc_test_f1', 0):.4f}"
          f" → Δ={acalb1_tf1 - acalb1_info.get('unc_test_f1', 0):+.4f}")
    print(row("B_b1. Gating (Beam-1)",     bb1_vp, bb1_vr, bb1_val, bb1_tp, bb1_tr, bb1_test))
    print("=" * 90)

    # ============================================================
    # 7) 【论文严格评估】用 tol=POSITION_TOLERANCE 重算 + Bootstrap 95% CI + 显著性检验
    # ============================================================
    print("\n" + "=" * 90)
    print("【论文严格评估】 tol=POSITION_TOLERANCE (与 build_candidates 对齐) + Bootstrap 95% CI")
    print("=" * 90)
    TOL_EVAL = POSITION_TOLERANCE  # 论文主评估 = 2 (与候选生成一致)

    def _label_list(samples):
        return [s['label_entities'] for s in samples]

    # 重算所有策略的 preds, 收集到 dict
    # val 列用 va_samples (来自 run_gating_fusion 内部 8:2 切出的 800 条, 门控网络真未见过的验证集)
    # test 列用 test_samples (1000 条, 最终测试)
    # 不用 train_samples (有数据泄露, 门控网络已在 tr_samples 上训过)
    strategy_preds = {}
    # 1) BERT Only
    bp_v_pre, _ = bert_only_predict(va_samples, train_llm, train_bert, source=LLM_SOURCE)
    bp_t_pre, _ = bert_only_predict(test_samples, test_llm, test_bert, source=LLM_SOURCE)
    strategy_preds['BERT Only']        = (bp_v_pre, bp_t_pre)
    # 2) LLM Only (Top-1, 论文主基线)
    lp_v_pre, _ = llm_only_predict(va_samples, beam_idx=0)
    lp_t_pre, _ = llm_only_predict(test_samples, beam_idx=0)
    strategy_preds['LLM Top-1']        = (lp_v_pre, lp_t_pre)
    # 3) A. Static Weight
    av_pre, _ = static_fusion_predict(va_samples, a_alphas, a_th,
        consensus_bonus=a_hyper['bonus'], consensus_th_mult=a_hyper['mult'],
        bert_only_factor=a_hyper['bf'], llm_only_factor=a_hyper['lf'],
        vote_bonus_coef=a_hyper['vbc'])
    at_pre, _ = static_fusion_predict(test_samples, a_alphas, a_th,
        consensus_bonus=a_hyper['bonus'], consensus_th_mult=a_hyper['mult'],
        bert_only_factor=a_hyper['bf'], llm_only_factor=a_hyper['lf'],
        vote_bonus_coef=a_hyper['vbc'])
    strategy_preds['A. Static Weight']  = (av_pre, at_pre)
    # 4) A''. Vote-Aware
    avv_pre, _ = static_fusion_voteaware_predict(va_samples, app_th, app_mvb, app_mvl)
    atv_pre, _ = static_fusion_voteaware_predict(test_samples, app_th, app_mvb, app_mvl)
    strategy_preds["A''. Vote-Aware"]  = (avv_pre, atv_pre)
    # 5) A_v2
    av_v2_pre, _ = a_v2_predict(va_samples, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d)
    at_v2_pre, _ = a_v2_predict(test_samples, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d)
    strategy_preds['A_v2. Consensus']  = (av_v2_pre, at_v2_pre)
    # 5b) A_cal: Calibrated Weighted Fusion (5-beam)
    # 关键: val 严格评估用 va_samples (8:2 切, 门控网络未见) + 5-beam 校准器
    #       test 用 test_samples + 5-beam 校准器
    #       A_cal_alphas/th 由 run_calibrated_weighted_fusion 内部在 train_samples
    #       拟合超参, 用同一 alphas/th 在 va_samples (门控网络未见, 800 条)
    #       上重算 score → 严格无数据泄露
    # 注: acal_train_cal / acal_test_cal 是 deep copy 后的校准样本, 不能直接用
    #     va_samples (会丢校准步骤). 解决: 在 strict 块重新校准 va_samples / test_samples
    av_cal_pre = _predict_with_calibrators_reuse_alphas(
        va_samples, train_samples, acal_alphas, acal_th)
    at_cal_pre = _predict_with_calibrators_reuse_alphas(
        test_samples, train_samples, acal_alphas, acal_th)
    strategy_preds['A_cal. Calibrated']  = (av_cal_pre, at_cal_pre)
    # 5b') A_cal_b1: Calibrated Weighted Fusion (Beam-1 only)
    # 关键: 严格评估必须用 b1 解析的样本 (va_samples_b1 / test_samples_b1),
    #       不可用 5b 解析的 va_samples / test_samples — 5b 的 llm_conf 已经是
    #       α·max+(1-α)·mean+β·(N-1) 聚合, 与 b1 的单 beam conf 分布不同,
    #       会导致 acalb1_alphas/th 应用到错误分布, F1 严重偏低
    av_calb1_pre = _predict_with_calibrators_reuse_alphas(
        va_samples_b1, train_samples_b1, acalb1_alphas, acalb1_th)
    at_calb1_pre = _predict_with_calibrators_reuse_alphas(
        test_samples_b1, train_samples_b1, acalb1_alphas, acalb1_th)
    strategy_preds['A_cal_b1. Calibrated (Beam-1)'] = (av_calb1_pre, at_calb1_pre)
    # 6) A_b1 (消融: Static Weight + Beam-1 only)
    # 关键: 严格评估必须用 b1 解析的样本 (va_samples_b1 / test_samples_b1),
    #       不可用 5b 解析的 va_samples / test_samples — 5b 的 llm_conf 已经是
    #       α·max+(1-α)·mean+β·(N-1) 聚合, 与 b1 的单 beam conf 分布不同,
    #       会导致 ab1_alphas/ab1_th 应用到错误分布, F1 严重偏低
    #       (旧版 val 0.7205 / test 0.7222 → 修复后应接近 main 表 0.7519/0.7483)
    av_b1_pre, _ = static_fusion_predict(va_samples_b1, ab1_alphas, ab1_th,
        consensus_bonus=ab1_hyper['bonus'], consensus_th_mult=ab1_hyper['mult'],
        bert_only_factor=ab1_hyper['bf'], llm_only_factor=ab1_hyper['lf'],
        vote_bonus_coef=ab1_hyper['vbc'])
    at_b1_pre, _ = static_fusion_predict(test_samples_b1, ab1_alphas, ab1_th,
        consensus_bonus=ab1_hyper['bonus'], consensus_th_mult=ab1_hyper['mult'],
        bert_only_factor=ab1_hyper['bf'], llm_only_factor=ab1_hyper['lf'],
        vote_bonus_coef=ab1_hyper['vbc'])
    strategy_preds['A_b1. Static (Beam-1)']  = (av_b1_pre, at_b1_pre)
    # 8) B. Gating Network
    if gating_model is not None:
        # 加速: 一次性批量推理拿 val/test scores, 然后按 best_type_th 筛
        bv_scores = _gating_batch_scores(va_samples, gating_model,
                                          drop=getattr(gating_model, 'drop_beam_features', False))
        bt_scores = _gating_batch_scores(test_samples, gating_model,
                                          drop=getattr(gating_model, 'drop_beam_features', False))
        bv_pre, _ = _filter_by_threshold(va_samples, bv_scores, b_th)
        bt_pre, _ = _filter_by_threshold(test_samples, bt_scores, b_th)
        strategy_preds['B. Gating Net'] = (bv_pre, bt_pre)
    else:
        print("  [skip B] gating_model 不在 main 作用域, 跳过 B 的严格评估")
    # 7) B_b1 (消融: Gating + Beam-1 only)
    # 关键: 严格评估必须用 b1 解析的样本 (bb1_va / test_samples_b1),
    #       不可用 5b 解析的 va_samples / test_samples — bb1_model 在 12 维通用特征上训练,
    #       应用到 5b 样本的 llm_conf 分布会失配, F1 严重偏低
    if bb1_model is not None:
        bv_b1_scores = _gating_batch_scores(bb1_va, bb1_model,
                                             drop=getattr(bb1_model, 'drop_beam_features', False))
        bt_b1_scores = _gating_batch_scores(test_samples_b1, bb1_model,
                                             drop=getattr(bb1_model, 'drop_beam_features', False))
        bv_b1_pre, _ = _filter_by_threshold(bb1_va, bv_b1_scores, bb1_th)
        bt_b1_pre, _ = _filter_by_threshold(test_samples_b1, bt_b1_scores, bb1_th)
        strategy_preds['B_b1. Gating (Beam-1)'] = (bv_b1_pre, bt_b1_pre)
    else:
        print("  [skip B_b1] bb1_model 不在 main 作用域, 跳过 B_b1 严格评估")

    val_labels = _label_list(va_samples)   # 门控网络未见的 800 条的 label (与训练集 tr_samples 互不重叠)
    test_labels = _label_list(test_samples)

    # 加速: 一次性预计算所有策略的 (tps, pns, tns), 避免 bootstrap 重算 set
    print(f"\n  [预计算] 8 策略 × (val, test) = 16 组 metrics, 一次性算 ...")
    pre_v = {name: _precompute_metrics(vp, val_labels, tol=TOL_EVAL)
             for name, (vp, _) in strategy_preds.items()}
    pre_t = {name: _precompute_metrics(tp, test_labels, tol=TOL_EVAL)
             for name, (_, tp) in strategy_preds.items()}

    # 用 tol=2 重算 P/R/F1
    print(f"\n  tol = {TOL_EVAL} (与 build_candidates 一致)")
    # 表头: val/test 各列 P / R / F1 + 95% CI
    print(f"  {'策略':<22} | {'val P':>7} {'val R':>7} {'val F1':>7} {'95% CI':<17} | "
        f"{'test P':>7} {'test R':>7} {'test F1':>7} {'95% CI':<17} | {'Δ F1':>8}")
    print("  " + "-" * 120)

    strict_results = {}
    for name, (vp, tp) in strategy_preds.items():
        tps_v, pns_v, tns_v = pre_v[name]
        tps_t, pns_t, tns_t = pre_t[name]
        # 整体 P/R/F1
        tp_sum = int(tps_v.sum()); pn_sum = int(pns_v.sum()); tn_sum = int(tns_v.sum())
        pv = tp_sum / pn_sum if pn_sum else 0.0
        rv = tp_sum / tn_sum if tn_sum else 0.0
        fv = 2 * pv * rv / (pv + rv) if (pv + rv) else 0.0
        tp_sum_t = int(tps_t.sum()); pn_sum_t = int(pns_t.sum()); tn_sum_t = int(tns_t.sum())
        pt = tp_sum_t / pn_sum_t if pn_sum_t else 0.0
        rt = tp_sum_t / tn_sum_t if tn_sum_t else 0.0
        ft = 2 * pt * rt / (pt + rt) if (pt + rt) else 0.0
        # Bootstrap
        mv, lov, hiv, _ = bootstrap_f1_ci_from_arr(tps_v, pns_v, tns_v, n_boot=1000)
        mt, lot, hit_, _ = bootstrap_f1_ci_from_arr(tps_t, pns_t, tns_t, n_boot=1000)
        strict_results[name] = (pv, rv, fv, pt, rt, ft, mv, lov, hiv, mt, lot, hit_)
        delta = ft - strict_results.get('BERT Only', (0, 0, 0, 0, 0, 0))[5]
        print(f"  {name:<22} | {pv:>7.4f} {rv:>7.4f} {fv:>7.4f} [{lov:>5.3f}, {hiv:>5.3f}] | "
            f"{pt:>7.4f} {rt:>7.4f} {ft:>7.4f} [{lot:>5.3f}, {hit_:>5.3f}] | {delta:>+8.4f}")

    # 关键对比的 paired bootstrap 显著性检验 (走预计算数组)
    print(f"\n  【显著性检验】Paired Bootstrap (H0: ΔF1 = 0, n_boot=1000):")
    base = 'BERT Only'
    if base in strategy_preds:
        for name in ['LLM Top-1', 'A. Static Weight',
                      "A''. Vote-Aware", 'A_v2. Consensus',
                      'A_cal. Calibrated',
                      'A_b1. Static (Beam-1)', 'A_cal_b1. Calibrated (Beam-1)',
                      'B_b1. Gating (Beam-1)', 'B. Gating Net']:
            if name not in strategy_preds:
                continue
            dm_v, lo_v, hi_v, pv_v = paired_bootstrap_pvalue_from_arr(
                *pre_v[base], *pre_v[name], n_boot=1000)
            dm_t, lo_t, hi_t, pv_t = paired_bootstrap_pvalue_from_arr(
                *pre_t[base], *pre_t[name], n_boot=1000)
            sig_v = "***" if pv_v < 0.001 else "**" if pv_v < 0.01 else "*" if pv_v < 0.05 else "ns"
            sig_t = "***" if pv_t < 0.001 else "**" if pv_t < 0.01 else "*" if pv_t < 0.05 else "ns"
            print(f"    {base} vs {name}: "
                  f"ΔF1 val = {dm_v:+.4f} [{lo_v:+.4f}, {hi_v:+.4f}] p={pv_v:.3f} {sig_v} | "
                  f"ΔF1 test = {dm_t:+.4f} [{lo_t:+.4f}, {hi_t:+.4f}] p={pv_t:.3f} {sig_t}")


if __name__ == "__main__":
    main()
