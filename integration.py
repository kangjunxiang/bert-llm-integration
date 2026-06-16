"""
LLM + BERT Entity Fusion
========================================
Implements three main fusion strategies and their ablations:
    Strategy A   Static Weight Fusion               (static-weight fusion, no training)
    Strategy A'' Vote-Aware Hard-Rule Static        (hard rule + vote gate, no training)
    Strategy A_v2 Consensus V2 Hard-Rule            (per-beam vote + conf spread, no training)
    Strategy A_cal Calibrated Weighted Fusion      (Isotonic calibration + per-type α weighting, no training)
    Strategy B   Gating Network Fusion              (gating network, end-to-end learning)
Ablation studies (Beam-1 only):
    Strategy A_b1    / Strategy A_cal_b1    / Strategy B_b1
Console output shows P/R/F1 of each strategy on the validation and test sets.
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
# Single global seed to keep experiments reproducible
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
# Pin the Python hash seed to avoid non-determinism from dict ordering
if not os.environ.get('PYTHONHASHSEED'):
    os.environ['PYTHONHASHSEED'] = '0'

# ====================================================================
# 0. Global constants
# ====================================================================
ALL_TYPES = ['dis', 'sym', 'dru', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep']
NUM_TYPES = len(ALL_TYPES)
TYPE2IDX = {t: i for i, t in enumerate(ALL_TYPES)}

# Position tolerance: predictions from LLM/BERT with |start diff| <= TOL are
# treated as referring to the same entity
POSITION_TOLERANCE = 2


# ====================================================================
# 1. Data parsing
# ====================================================================
def _safe_parse_json(line: str):
    """Safely parse JSON; fall back to ast.literal_eval on failure
    (supports single-quoted strings)."""
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        try:
            return ast.literal_eval(line.strip())
        except Exception:
            return None


def _is_mark_format_line(line: str) -> bool:
    """Detect whether a single line uses the mark annotation format
    (the predict field contains the ' :' marker).

    Mark format signature: the line contains ' :' flanked by square brackets,
    e.g. "[entity] :type".
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
    """Sanitize an entity type, returning a valid type or None."""
    VALID = set(ALL_TYPES)
    if etype in VALID:
        return etype
    cleaned = etype.lstrip('-.、。,，)）:： ')
    return cleaned if cleaned in VALID else None


def _split_mark_line(line):
    """Split a line of the form 'etype : content' into (etype, content);
    return None on failure."""
    line = line.strip()
    if not line or ' :' not in line:
        return None
    parts = line.split(' :', 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1]


def _entities_with_pos_in_content(content):
    """Extract (start, end, entity) triples from a single mark-format content
    line. start/end are in-line offsets after stripping the surrounding [],
    and entity is already trimmed.

    Two input forms are supported:
      A) Original sentence with [entity] markers  → exact offsets
      B) Bare entities: 'fever_A, fever_B' (or Chinese terms such as
         '稽留热、弛张热' for medical NER) → split on [、,;；], offsets
         are positions within the content
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
    """Parse mark-format text into a list of
    (start, end, entity, type, line_idx) tuples.

    Positions are real indices inside the line content after stripping [ and ],
    closed interval (start + len(entity) - 1).
    Both the [entity] marker form and the bare-entity form are supported.
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
    """Realign mark-format positions to their true indices inside the
    full `original_text`.

    Rules:
      1. Entity not present in original_text → drop (LLM hallucination, not
         counted in evaluation).
      2. Entity present, position already correct → keep as is.
      3. Entity present, position mismatched:
         a. Exactly one occurrence → use that occurrence.
         b. Multiple occurrences:
            - keep_all_matches=False (default): within `tolerance` of the
              predicted start, pick the closest occurrence; if none qualifies,
              fall back to the closest occurrence overall.
            - keep_all_matches=True:  keep every occurrence as a candidate.
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
    """Mark-format predict → entity_dict:
    {(start, end, type, entity): {conf, entity, type, start, end}}"""
    ents = parse_marked_text_with_pos(predict_text)
    ents = _correct_entity_positions(ents, original_text)
    out = {}
    for s, e, txt, t, _ in ents:
        key = (s, e, t, txt)
        if key not in out:
            out[key] = {'confidence': default_conf, 'entity': txt, 'type': t,
                        'start_idx': s, 'end_idx': e}
    return out


# LLM multi-beam aggregation parameters (max+mean+vote_reward model)
# Formula:  P = α · max(c_i) + (1-α) · mean(c_i) + β · (N-1)
#  - α: weight for the best-beam confidence (default 0.7)
#  - β: linear reward for the number of agreeing beams (default 0.05)
#  - N: number of beams that predicted this entity
LLM_ALPHA = None      # type: float | None
LLM_VOTE_REWARD = None  # type: float | None


def _aggregate_llm_conf(confs, beam_idxs, alpha, beta):
    """Aggregate the confidences of multiple beams that predicted the same
    entity using the max+mean+vote_reward rule.

    P = α · max(c_i) + (1-α) · mean(c_i) + β · (N-1), clamped to [0, 1]
    """
    if not confs:
        return 0.0
    a = alpha if alpha is not None else 0.7
    b = beta if beta is not None else 0.05
    score = a * max(confs) + (1 - a) * (sum(confs) / len(confs)) + b * (len(confs) - 1)
    return max(0.0, min(1.0, score))


def _parse_mark_format_beam(models, original_text, default_conf=0.8):
    """max+mean+vote_reward aggregation over multiple beams.

    Input:  models = [ {predict: str, confidence: float}, ... ]
            (ranked 1→N by beam)
    Output: { (start, end, type, text): {confidence, vote_count, best_beam_idx,
              best_beam_conf, beam_idxs, beam_confs, llm_avg_conf,
              llm_max_conf, ...} }
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
    """Mark-format label → [{entity, start_idx, end_idx, type}, ...]"""
    return [
        {'entity': txt, 'start_idx': s, 'end_idx': e, 'type': t}
        for s, e, txt, t, _ in parse_marked_text_with_pos(label_text)
    ]


def _normalize_entity_confidences(entity_dict):
    """Normalize confidences into [0, 1]; leave them untouched when all values
    are the same to avoid injecting random noise."""
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
    """Parse one LLM file line →
    (text, entity_dict, label_entities, llm_parse_error).

    use_beam=True:  vote over multiple beams; entity_dict carries
                    vote_count / llm_avg_conf / llm_max_conf.
    use_beam=False: use only the first beam; c['llm_conf'] is the true
                    Beam-1 confidence, vote_count=1.
    """
    obj = _safe_parse_json(line)
    if obj is None:
        return None
    text = obj['text']
    models = obj.get(source, []) or obj.get('ner_models', [])

    # Parse the label as the ground truth
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
            # Single-beam path also goes through the beam aggregator so that
            # the candidate carries the real conf plus vote_count=1,
            # keeping a unified interface.
            entity_dict = _parse_mark_format_beam(models[:1], text, default_conf=default_conf)
        entity_dict = _normalize_entity_confidences(entity_dict)
        return text, entity_dict, label_entities, obj.get('llm_parse_error', False)

    # Standard JSON format
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
    """Parse one BERT file line → (text, [entity dicts])
    BERT's start_idx = label position + 1, so we subtract 1 to align with LLM.
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
# 2. Candidate generation / matching
# ====================================================================
def _entities_match(e1, e2, tol=0):
    """Same type + same text + |start diff| <= tol → considered the same entity."""
    if e1.get('entity', '') != e2.get('entity', ''):
        return False
    if e1.get('type', '') != e2.get('type', ''):
        return False
    if abs(int(e1.get('start_idx', 0)) - int(e2.get('start_idx', 0))) > tol:
        return False
    return True


def _entity_key(e):
    """(start_idx, entity_text, type) — used as candidate key / for evaluation."""
    return (e['start_idx'], e['entity'], e['type'])


def build_candidates(text, entity_dict, bert_entities, tol=0):
    """Merge LLM entities (entity_dict) with BERT entities and emit a candidate
    list.

    Each candidate: {start_idx, end_idx, type, entity, llm_conf, bert_conf,
                     llm_present, bert_present, [optional beam features] ...}
    Merge rule: prefer merging on the same (start, entity, type); otherwise,
    (same type, same entity with |start diff| <= tol) is also considered the
    same candidate, and BERT matches the closest LLM candidate.
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
    """Build the label key set (with position tolerance expansion)."""
    if tol <= 0:
        return {(int(e['start_idx']), e.get('entity', ''), e.get('type', ''))
                for e in label_entities if 'start_idx' in e}
    keys = set()
    for e in label_entities:
        for off in range(-tol, tol + 1):
            keys.add((int(e['start_idx']) + off, e['entity'], e['type']))
    return keys


def _entities_match_in_list(target, candidates, tol=0):
    """Find the first entity in `candidates` that matches `target`."""
    for c in candidates:
        if _entities_match(target, c, tol):
            return c
    return None


def _reaggregate_llm_confs(samples, alpha, beta):
    """Recompute each candidate's llm_conf under the new (α, β).

    Reuses the beam_confs already stored in each candidate; no re-parsing.
    Falls back to a 2-point approximation ([llm_max_conf, llm_conf]) when
    beam_confs is missing.
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
# 3. Evaluation
# ====================================================================
def evaluate(preds_list, labels_list, tol=0):
    """startIndexMatch evaluation: count hits of the
    (start_idx, entity, type) triple set.
    tol: position tolerance (paper main evaluation: 2).
    """
    tp, total_pred, total_true = _precompute_metrics(preds_list, labels_list, tol)
    tp = int(tp.sum()); total_pred = int(total_pred.sum()); total_true = int(total_true.sum())
    p = tp / total_pred if total_pred else 0.0
    r = tp / total_true if total_true else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, total_pred, total_true


def _precompute_metrics(preds_list, labels_list, tol=0):
    """Precompute (tp, |p_set|, |l_set|) for each sample → returns 3 np.arrays.

    Used by bootstrap resampling for vectorized sums (no set-intersection
    recomputation).
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
    """Paired bootstrap for the F1 95% confidence interval.

    Method: resample text indices with replacement n_boot times, compute F1
    for each, and take the [α/2, 1-α/2] percentiles.
    Returns: (mean, lo, hi, std)
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
    """Paired bootstrap test of H0: F1(B) - F1(A) = 0.

    Returns: (delta_mean, delta_ci_lo, delta_ci_hi, p_value_two_sided)
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
    """Array version of paired_bootstrap_pvalue: precomputed arrays can be
    shared across multiple strategies."""
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
    # Two-sided p-value: H1: ΔF1 ≠ 0
    n_le = int((deltas <= 0).sum())
    n_gt = n_boot - n_le
    p_val = float(min(n_le, n_gt) * 2 / n_boot)
    p_val = min(p_val, 1.0)
    return float(deltas.mean()), float(lo), float(hi), p_val


def bootstrap_f1_ci_from_arr(tps, pns, tns, n_boot=1000, alpha=0.05,
                              seed=GLOBAL_SEED):
    """Array version of bootstrap_f1_ci: precomputed arrays can be shared
    across multiple strategies."""
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
    """Per-type F1 of a single model.
    source='bert' → only BERT-only candidates; predict when bert_conf >= conf_th.
    source='llm'  → only LLM-only + consensus candidates; predict when
                    llm_conf >= conf_th.
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
    """Simulate LLM using only the beam_idx-th beam and compute P/R/F1.

    Note: ground truth must come from the LLM file's `label` field
    (manual annotation), not from the BERT file's `entities`.
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
# 4. Data loading & splitting
# ====================================================================
def load_data(llm_path, bert_path):
    with open(llm_path, 'r', encoding='utf-8') as f1, \
         open(bert_path, 'r', encoding='utf-8') as f2:
        llm_lines = f1.readlines()
        bert_lines = f2.readlines()
    n = min(len(llm_lines), len(bert_lines))
    return llm_lines[:n], bert_lines[:n]


def split_train_test(llm_lines, bert_lines, test_size=2000, seed=GLOBAL_SEED):
    """Split out the test set by test_size; remainder is the train set.
    Uses the given seed for reproducible splits."""
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
    """Generate (text, candidates, label_set) triples for every line; used to
    train the gating network / for evaluation.

    use_beam=True:  When parsing the LLM output, use 5-beam voting; the
        generated candidates carry extra features such as vote_count.
    Ground truth always comes from the `label` field of the LLM file
    (human-annotated).
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
# 5. Strategy A: Static Weight Fusion
# ====================================================================
# Idea: no training. Compute BERT-only F1 and LLM-only F1 on the validation
# set, then use the ratio as the weight:
# α = bert_F1 / (bert_F1 + llm_F1),  score = α*bert_conf + (1-α)*llm_conf
# A global threshold `th` is grid-searched, then refined per type.
def bert_only_predict(samples, llm_lines, bert_lines, source='ner_models'):
    """BERT Only baseline: only keep the entities predicted by BERT."""
    preds_list, labels_list = [], []
    for s in samples:
        bert_pred = [c for c in s['candidates'] if c['bert_present']]
        preds_list.append(bert_pred)
        labels_list.append(s['label_entities'])
    return preds_list, labels_list


def llm_only_predict(samples, llm_lines=None, bert_lines=None,
                     source='ner_models', beam_idx=None, conf_th=None):
    """LLM Only baseline: only keep the entities predicted by the LLM.

    Parameters:
        beam_idx: int | None
            None  -> 5-beam union (no threshold; position-deduped)  ← ablation
            0..4  -> only use the beam_idx-th beam                  ← paper main
                     baseline (LLM Top-1)
        conf_th: float | None
            None   -> no confidence filtering
            0.0~1  -> keep only entities with best_beam_conf >= conf_th
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
    """Per-type entity-level F1 of BERT-only and LLM-only on the validation set."""
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
    """Statistics: precision of consensus (BERT+LLM agree) vs single-source
    predictions. Used as the basis for the consensus bonus in Strategy A."""
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
    """Static weight fusion (3-case scoring).

    Core idea: consensus entities have much higher precision than single-source
    predictions, so they should get a sizeable bonus and a lower threshold.
        both-present: score = α*bert + (1-α)*llm + bonus + vote_bonus_coef * log(vc)/log(5),  th *= consensus_th_mult
        bert-only:    score = bert_only_factor * bert_conf,  th unchanged
        llm-only:     score = llm_only_factor  * llm_conf,   th unchanged
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
    """Fast version of static_fusion_predict: returns (tps, pns, tns) np.arrays
    directly, without building a list of dicts.

    Uses s['label_set'] (already a set) for O(1) hit checks, ~5-10× faster
    than `evaluate`. Used to accelerate the 6000-cell global grid in
    `run_static_fusion`.
    """
    n = len(samples)
    tps = np.zeros(n, dtype=np.int64)
    pns = np.zeros(n, dtype=np.int64)
    tns = np.empty(n, dtype=np.int64)
    for i, s in enumerate(samples):
        lset = s['label_set']  # Already tolerance-expanded, used for O(1) hit checks
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
        # Note: label_set is already tolerance-expanded; do NOT use len(lset)
        # directly — use the true label count to match `evaluate`.
        tns[i] = len(s['label_entities'])
    return tps, pns, tns


def _f1_from_arr(tps, pns, tns):
    """Compute P/R/F1 from the (tps, pns, tns) arrays — used by the fast
    version of `evaluate`."""
    tp = int(tps.sum())
    pn = int(pns.sum())
    tn = int(tns.sum())
    p = tp / pn if pn else 0.0
    r = tp / tn if tn else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def run_static_fusion(train_samples, test_samples):
    """Strategy A: Static Weight Fusion (3-case, with consensus bonus)."""
    print("\n" + "=" * 70)
    print("【Strategy A】Static Weight Fusion (3-case)")
    print("=" * 70)
    print("Idea: consensus entities have much higher precision than single-source.")
    print("      - both-present: score = α*bert + (1-α)*llm + bonus, threshold × mult")
    print("      - bert-only:    score = bert_conf")
    print("      - llm-only:     score = llm_conf")
    print("      Grid search: α (per-type) + bonus / mult / factor (global) + threshold (per-type)")
    print("-" * 70)

    # 1) Per-type F1 on the validation set (used for α) + consensus diagnosis
    bert_f1, llm_f1 = per_type_f1(train_samples)
    cs = _consensus_stats(train_samples)
    print("  [1] Validation per-model F1 + consensus diagnosis:")
    print(f"      {'type':<6} {'BERT F1':>9} {'LLM F1':>9} {'α':>8}")
    for t in ALL_TYPES:
        b, l = bert_f1[t], llm_f1[t]
        denom = b + l
        a = b / denom if denom > 0 else 0.5
        print(f"      {t:<6} {b:>9.4f} {l:>9.4f} {a:>8.4f}")
    print(f"      consensus both_prec  = {cs['both_prec']:.4f}  (n={cs['both_n']})")
    print(f"      bert_only_prec  = {cs['bert_prec']:.4f}  (n={cs['bert_only_n']})")
    print(f"      llm_only_prec   = {cs['llm_prec']:.4f}  (n={cs['llm_only_n']})")

    # 2) Global hyperparameter search: consensus_bonus, consensus_th_mult,
    #    bert_only_factor, llm_only_factor
    #    + vote_bonus_coef (5-beam vote bonus)
    #    α is fixed per type from the F1 ratio; thresholds are grid-searched
    #    per type.
    #    Objective: F1 - 0.02 * |P - R|  (P/R balance, anti-overfit)
    print("\n  [2] Global hyperparameter grid search (on the validation set, "
          "objective F1 - 0.02·|P-R|):")
    best = {'f1': 0.0, 'score': -1.0}
    type_alphas = {t: (bert_f1[t] / (bert_f1[t] + llm_f1[t])
                       if (bert_f1[t] + llm_f1[t]) > 0 else 0.5) for t in ALL_TYPES}
    # Speedup: use static_fusion_predict_metrics (fast, returns tps/pns/tns directly)
    # Search space: bonus×mult×bf×lf×vbc×th. β→vbc was already searched in
    # Step 1.5, so vbc=0 (best) is fixed here. bonus/bf/lf are narrowed to the
    # empirically effective range → 1080 cells (was 7200, 6.7× speedup).
    _bonus_grid = [0.05, 0.15, 0.25]
    _mult_grid = [0.7, 0.85, 1.0]
    _bf_grid = [0.9, 1.0]
    _lf_grid = [0.7, 0.85, 1.0]
    _vbc_grid = [0.00]  # β was already searched in Step 1.5, so vbc=0 is fixed
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
                                print(f"      [Progress] {_cnt}/{_total}  "
                                      f"elapsed {_elapsed:.1f}s, ETA {_eta:.1f}s, "
                                      f"current best F1={best['f1']:.4f}")
    print(f"  -> Global best: bonus={best['bonus']}, mult={best['mult']}, "
          f"bf={best['bf']}, lf={best['lf']}, th={best['th']}, "
          f"vbc={best['vbc']}, F1={best['f1']:.4f} (P={best['p']:.4f}, R={best['r']:.4f})")

    # 3) Per-type threshold refinement (fast)
    print("\n  [3] Per-type threshold refinement:")
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

    # 4) Final on the validation set
    print("\n  [4] Final results on the validation set:")
    val_preds, val_labels = static_fusion_predict(
        train_samples, type_alphas, best_type_th,
        consensus_bonus=best['bonus'], consensus_th_mult=best['mult'],
        bert_only_factor=best['bf'], llm_only_factor=best['lf'],
        vote_bonus_coef=best['vbc'])
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels)
    print_metrics("Static-Weight (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    # 5) Test set
    test_preds, test_labels = static_fusion_predict(
        test_samples, type_alphas, best_type_th,
        consensus_bonus=best['bonus'], consensus_th_mult=best['mult'],
        bert_only_factor=best['bf'], llm_only_factor=best['lf'],
        vote_bonus_coef=best['vbc'])
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels)
    print_metrics("Static-Weight (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)
    # Returns (F1_val, F1_test, type_alphas, type_thresholds, hyperparam dict)
    hyper = {
        'bonus': best['bonus'], 'mult': best['mult'],
        'bf': best['bf'], 'lf': best['lf'], 'vbc': best['vbc'],
    }
    return val_f1, test_f1, type_alphas, best_type_th, hyper


# ====================================================================
# 5b. Strategy A'' (Vote-Aware Hard-Rule version): encode the beam vote
# count explicitly into the threshold
# ====================================================================
# On top of A', two global parameters are added:
#   min_vote_both: minimum beams for consensus (BERT+LLM agree)
#   min_vote_llm : minimum beams for LLM-only (LLM-only often hallucinates,
#                  default requires 5 votes)
# Intuition:
#   - 5-vote LLM-only = all 5 beams predict it but BERT does not → maybe
#                      the LLM learned something BERT missed
#   - 1-vote consensus = 1 beam agrees + BERT agrees            → weak signal
#   - 5-vote consensus = full agreement                          → strong signal


def _make_uniform_type_th(con_th, b_th, l_th):
    """Build a type_th dict: {type: (con_th, b_th, l_th)}; all types share
    the same threshold triple."""
    return {t: (con_th, b_th, l_th) for t in ALL_TYPES}


def static_fusion_voteaware_predict(samples, type_th, min_vote_both=1, min_vote_llm=5):
    """Vote-Aware hard-rule fusion.
    type_th[t] = (con_th, b_th, l_th)
    Rules:
        consensus:     vote_count >= min_vote_both AND bert_conf >= con_th[t]
        BERT-only:     bert_conf >= b_th[t]
        LLM-only:      vote_count >= min_vote_llm  AND llm_conf  >= l_th[t]
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
    """Strategy A'': Vote-Aware Hard-Rule Static Fusion."""
    print("\n" + "=" * 70)
    print("【Strategy A''】Vote-Aware Hard-Rule Static Fusion (hard rule + vote gate)")
    print("=" * 70)
    print("Idea: on top of A', encode the beam vote count explicitly:")
    print("      - consensus: vote_count >= min_vote_both AND bert_conf >= con_th")
    print("      - LLM-only:  vote_count >= min_vote_llm  AND llm_conf  >= l_th")
    print("      LLM-only is often hallucination, so default requires 5/5 votes.")
    print("-" * 70)

    # 1) Global grid: min_vote_both (min votes for consensus) × min_vote_llm
    #    (min votes for LLM-only) × bsolo (BERT solo threshold) × lsolo
    #    (LLM solo threshold, con=0 fixed)
    print("  [1] Global grid (min_vote_both × min_vote_llm × bsolo × lsolo):")
    best = {'f1': 0.0}
    for mvb in [1, 2, 3, 4, 5]:                       # min votes for consensus
        for mvl in [3, 4, 5]:                         # min votes for LLM-only
            for bsolo in [0.80, 0.85, 0.90, 0.95]:
                for lsolo in [0.80, 0.85, 0.90, 0.95]:
                    type_th = _make_uniform_type_th(0.0, bsolo, lsolo)
                    preds, labels = static_fusion_voteaware_predict(
                        train_samples, type_th, mvb, mvl)
                    _, _, f1, _, _, _ = evaluate(preds, labels)
                    if f1 > best['f1']:
                        best = {'f1': f1, 'mvb': mvb, 'mvl': mvl,
                                'bsolo': bsolo, 'lsolo': lsolo}
    print(f"  -> Global best: min_vote_both={best['mvb']}, min_vote_llm={best['mvl']}, "
          f"bsolo={best['bsolo']}, lsolo={best['lsolo']}, F1={best['f1']:.4f}")

    # 2) Per-type consensus_th refinement (it has the most impact; the
    #    shared solo thresholds keep the global value)
    print("\n  [2] Per-type consensus_th refinement:")
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

    # 3) Per-type solo threshold refinement (around the global value)
    print("\n  [3] Per-type bert_solo_th / llm_solo_th refinement:")
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

    # 4) Per-type min_vote_both refinement (around the global)
    print("\n  [4] Per-type min_vote_both refinement (LLM-only votes share the global):")
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
        # Note: min_vote_both is global at inference; this loop only verifies
        # the global value's stability across types.
    # All types use the best global mvb / mvl
    mvb_final = best['mvb']
    mvl_final = best['mvl']
    print(f"      All types use the global: min_vote_both={mvb_final}, min_vote_llm={mvl_final}")

    # 5) Validation
    print("\n  [5] Final results on the validation set:")
    val_preds, val_labels = static_fusion_voteaware_predict(
        train_samples, type_th, mvb_final, mvl_final)
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels)
    print_metrics("Vote-Aware Hard-Rule (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    # 6) Test
    test_preds, test_labels = static_fusion_voteaware_predict(
        test_samples, type_th, mvb_final, mvl_final)
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels)
    print_metrics("Vote-Aware Hard-Rule (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)
    return val_f1, test_f1, type_th, mvb_final, mvl_final


# ====================================================================
# 6. Strategy B: Gating Network Fusion
#   - The network outputs two sigmoid weights (w_bert, w_llm).
#   - Combined score = w_bert * bert_conf + w_llm * llm_conf +
#                      consensus_bonus * both_present
#   - Training objective: push the combined score up when the entity is
#     correct, down otherwise (BCE on the combined score).
#   - The consensus feature (both_present) is fed to the network and is
#     also used as an explicit bonus at inference.
#   - The network learns: consensus  → both w_b and w_l high;
#                         single      → lean toward the high-confidence side.
# ====================================================================

class GatingNetwork(nn.Module):
    """Outputs two weights (w_bert, w_llm); the network learns "when to trust
    whom".

    n_feats: number of input features. Full = 16 (including 4 beam-specific
             ones); drop_beam = 12 (only generic features).

    Two architectures:
      - 5b mode (n_feats=16): 2×hidden=64 + 3 heads (w_bert / w_llm / bonus)
                              — same as the original
      - b1 mode (n_feats=12): 1×hidden=32 + 1 head that outputs a single logit
                              — aligned with gating_network_dp_mark_v3.py
                              Combined with BCE + 0.15 * MSE(calibrated_target)
                              distillation loss to mitigate val/test overfit.
    """
    def __init__(self, num_types=NUM_TYPES, type_emb_dim=8, hidden_dim=64, n_feats=16):
        super().__init__()
        self.n_feats = n_feats
        self.drop_beam_features = (n_feats == 12)  # remembered for gating_predict
        self.type_emb = nn.Embedding(num_types, type_emb_dim)
        if n_feats == 12:
            # b1 mode: small capacity + single head, to avoid overfit on the
            # small Beam-1 sample
            self.trunk = nn.Sequential(
                nn.Linear(12 + type_emb_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.head_score = nn.Linear(32, 1)
        else:
            # 5b mode: full 3 heads, same as the original
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
            # head 1: w_bert
            self.head_wb = nn.Linear(hidden_dim, 1)
            # head 2: w_llm
            self.head_wl = nn.Linear(hidden_dim, 1)
            # head 3: consensus bonus (sigmoid output multiplied by both_pres)
            self.head_bonus = nn.Linear(hidden_dim, 1)

    def forward(self, llm_conf, bert_conf, llm_pres, bert_pres,
                conf_diff, both_pres, llm_only, bert_only,
                n_cands, bert_rank, llm_rank, static_score,
                type_ids, *beam_feats):
        """Argument order: 12 base + type_ids + N beam_feats (0 or 4)
        Full mode (drop=False): beam_feats receives 4 tensors
        Drop mode (drop=True):  beam_feats is an empty tuple and is skipped
        """
        type_e = self.type_emb(type_ids)
        base = [llm_conf, bert_conf, llm_pres, bert_pres,
                conf_diff, both_pres, llm_only, bert_only,
                n_cands, bert_rank, llm_rank, static_score]
        x = torch.stack(base + list(beam_feats), dim=1)
        x = torch.cat([x, type_e], dim=1)
        h = self.trunk(x)
        if self.n_feats == 12:
            # b1 mode: directly return a single logit (sigmoid → 0..1 score)
            return self.head_score(h).squeeze(-1)
        # 5b mode: 3 heads
        w_bert_logit = self.head_wb(h).squeeze(-1)
        w_llm_logit  = self.head_wl(h).squeeze(-1)
        bonus_logit  = self.head_bonus(h).squeeze(-1)
        return w_bert_logit, w_llm_logit, bonus_logit


def _candidates_to_features(samples, drop_beam_features=False):
    """Split samples into (cand_features, type_ids, group_ids, labels) lists.
    drop_beam_features=True: do NOT feed the 5-beam-specific features
                             (vote_count / llm_avg_conf / llm_max_conf /
                             best_beam_idx); force the gating network to use
                             only the generic features (used by the B_b1
                             ablation).
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
                # B_b1 ablation: do NOT feed the 5-beam-specific features at
                # all; force the use of generic features only.
                # Also compute calibrated_target (0.5*llm + 0.5*bert) for the
                # distillation loss — aligned with v3.
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
                # Use the beam features if available; default to 0.0 (single-beam view)
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
    """Return order: 12 base + type_ids + (4 beam?) + group_ids + (calib?) + label
    Full mode  (drop=False): 12 + 1 + 4 + 1 + 0 + 1 = 19 tensors
    Drop mode   (drop=True):  12 + 1 + 0 + 1 + 1 + 1 = 16 tensors
                              (calib=0.5*llm+0.5*bert is fed for distillation)
    forward signature: 12 base + type_ids + 4 beam (positional, so model(*xs)
    must receive them in the same order).
    """
    if not feats:
        n_base = 12
        n_beam = 0 if drop_beam_features else 4
        empty_f = torch.empty(0, dtype=torch.float32)
        empty_l = torch.empty(0, dtype=torch.long)
        # Tensor order: n_base floats + 1 long (type) + n_beam floats + 1 long (group) + (1 float calib) + 1 float (label)
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
    tensors.append(torch.tensor(type_ids, dtype=torch.long))   # type_ids immediately after 12 base
    if not drop_beam_features:
        tensors += [torch.tensor([f[k] for f in feats], dtype=torch.float32) for k in beam_keys]
    tensors.append(torch.tensor(group_ids, dtype=torch.long))
    if drop_beam_features:
        # b1 mode: append calibrated_target (distillation soft label), between
        # group and label
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
    """Combine the network logits into a scalar "entity-correctness score"
    (used for training and inference)."""
    wb = torch.sigmoid(w_bert)
    wl = torch.sigmoid(w_llm)
    b  = torch.sigmoid(bonus_logit)
    # Consensus bonus only takes effect when both_present=1
    bonus_term = b * both_pres * 0.30
    return wb * bert_conf + wl * llm_conf + bonus_term, wb, wl, b


def _bce_loss_with_ranking(combined, labels, group_ids, margin=0.05):
    bce = nn.functional.binary_cross_entropy_with_logits(combined, labels)
    rank = _grouped_ranking_loss(combined, labels, group_ids, margin=margin)
    # Ranking is dominant (0.3); BCE is auxiliary (1.0) — mitigates a
    # too-conservative threshold
    return bce + 0.3 * rank


def train_gating(model, train_loader, valid_loader, epochs=50, lr=2e-3, patience=10,
                 save_dir='saved_models_clean', model_tag='5b'):
    """Train the gating network; loss is BCE + ranking on the combined score.
    model_tag: '5b' (5-beam features, 16-d) or 'b1' (Beam-1 only, 12-d) — this
               only decides the saved best-model filename, to keep the two
               ablations from overwriting each other.
    b1 mode: lr=5e-4 / wd=1e-3 / patience=15; loss = BCE + 0.15·MSE(calibrated_target)
             (aligned with gating_network_dp_mark_v3.py).
    """
    os.makedirs(save_dir, exist_ok=True)
    is_b1 = (model_tag == 'b1')
    # b1 mode uses more conservative hyperparameters (small lr + small wd +
    # long patience) — matches v3
    # Note: b1's lr is supplied by the caller (explicit 5e-4); wd/patience are
    # forced here so that old callers that pass 5 are not silently ignored.
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
    # One file per model_tag: 5b / b1 are saved separately to avoid overwrite
    best_path = os.path.join(save_dir, f'best_gating_{model_tag}.pth')
    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        if is_b1:
            # b1 loader layout: *xs, gid, calib, y
            for *xs, gid, calib, y in train_loader:
                optimizer.zero_grad()
                score_logit = model(*xs)            # single logit
                prob = torch.sigmoid(score_logit)
                bce = nn.functional.binary_cross_entropy_with_logits(score_logit, y)
                mse = nn.functional.mse_loss(prob, calib)
                loss = bce + 0.15 * mse
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
        else:
            # 5b loader layout: *xs, gid, y
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
    """Predict with the trained gating network.
    score = sigmoid(w_bert)*bert_conf + sigmoid(w_llm)*llm_conf + sigmoid(bonus)*both_pres*0.30
    threshold: float (global) or {type: float} (per-type)
    Note: model.drop_beam_features (set at training time) determines whether
    to drop the 5-beam features at inference.
    """
    drop = getattr(model, 'drop_beam_features', False)
    model.eval()
    # One batched inference for all samples' candidates, avoiding per-sample
    # calls in the loop
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
    """Compute the combined score for every candidate across all samples in a
    single forward pass (vectorized inference).
    Returns: list[np.array] — the s-th array contains the scores of all
    candidates of samples[s].
    Reused by the grid search: 1 batched inference + N threshold sweeps
    (N=40 global + 90 per-type).
    b1 mode: model outputs a single logit; score = sigmoid(logit)
    5b mode: model outputs 3 heads; score = wb*bert + wl*llm + bonus*both*0.30
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
    # The last 2 elements (or 3 in b1) are (gid, label) or (gid, calib, label).
    # Here we rebuild the tensor list with dummy labels to avoid ambiguity.
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
            # b1 mode: single logit → sigmoid
            scores = torch.sigmoid(out).cpu().numpy()
        else:
            # 5b mode: 3 heads → combined
            wb_log, wl_log, bonus_log = out
            combined, _, _, _ = _gating_combined(wb_log, wl_log, bonus_log, x[1], x[0], x[5])
            scores = combined.cpu().numpy()
    return [scores[start:end].astype(np.float32, copy=False)
            for (start, end) in boundaries]


def _filter_by_threshold(samples, scores_per_sample, threshold):
    """Filter by threshold (per-type dict or global float) on precomputed
    scores. Pure-Python loop that matches `gating_predict`'s output format,
    but does not call the model.
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
# 5c. Strategy A_v2: Consensus V2 Hard-Rule
# ====================================================================
# Granular consensus (per-beam features):
#   Strong consensus: best_beam_idx=0 (beam 1 also predicted) + vote >= 4
#                     → always keep
#   Weak consensus:   best_beam_idx >= 3 (only beams 4-5 predicted)
#                     → require an additional bert_conf threshold
#   Mid consensus:    default                                    → always keep
#
#   LLM-only:   vote >= 5  AND  max_beam_conf >= th_llm
#                          AND  (max-min) <= 0.30   ← conf-spread constraint
#
# Input parameters (per-type):
#   bsolo_th[type]          BERT-only bert_conf threshold
#   lsolo_th[type]          LLM-only max_beam_conf threshold
#   lsolo_min_vote[type]    LLM-only minimum vote count (default 5)
#   lsolo_max_spread[type]  LLM-only conf-spread upper bound (default 0.30)
def _a_v2_accept(c, bsolo, lsolo, lsolo_min_vote, lsolo_max_spread):
    """Whether a single candidate is accepted by A_v2 (based on per-beam
    vote + conf spread). Takes per-type threshold dicts and a candidate;
    returns True/False.
    """
    t = c['type']
    if c['bert_present'] and c['llm_present']:
        # === Consensus (BERT+LLM both predicted) ===
        # Weak consensus: only beams 4-5 predicted, BERT confidence is also
        # low → require bert_conf >= 0.50
        # Mid/strong consensus: always keep
        best_bi = c.get('best_beam_idx', 0)
        if best_bi >= 3:
            return c['bert_conf'] >= 0.50
        return True
    if c['bert_present'] and not c['llm_present']:
        # === BERT-only ===
        return c['bert_conf'] >= bsolo.get(t, 0.95)
    if c['llm_present'] and not c['bert_present']:
        # === LLM-only ===
        # Vote count + conf spread
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
    """A_v2 prediction: per-beam vote + conf spread + granular consensus.
    bsolo_th / lsolo_th : {type: float}
    lsolo_min_vote      : {type: int}    (default 5)
    lsolo_max_spread    : {type: float}  (default 0.30)
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
# Strategy A_v2 training / entry point
# ====================================================================
def run_a_v2(train_samples, test_samples):
    """A_v2: Consensus V2 Hard-Rule Fusion.
    Pipeline:
      [1] Global search: bsolo / lsolo / lsolo_min_vote / lsolo_max_spread
      [2] Per-type refinement (narrowed range, anti-overfit)
      [3] Output val / test F1
    """
    print("\n" + "=" * 70)
    print("【Strategy A_v2】Consensus V2")
    print("=" * 70)
    print("Granular consensus (bi>=3 adds bert_conf) + LLM-only conf-spread constraint")
    print("-" * 70)

    # [1] Diagnosis: mid/strong vs weak consensus
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
    print(f"  [1] Consensus diagnosis (train):")
    print(f"      All consensus:    n={both_n:5d}  hit={both_hit:5d}  prec={both_hit/both_n:.4f}")
    print(f"      Weak consensus (bi>=3): n={both_weak_n:5d}  hit={both_weak_hit:5d}  prec={both_weak_hit/max(1,both_weak_n):.4f}")

    # [2] Global hyperparameter grid search
    print(f"\n  [2] Global grid search (bsolo, lsolo, min_vote, max_spread):")
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
    print(f"  -> Global best: bsolo={bsolo}, lsolo={lsolo}, min_vote={mv}, max_spread={sp}, "
          f"F1={best_f1:.4f}")

    # [3] Per-type refinement (narrow range, anti-overfit)
    print(f"\n  [3] Per-type lsolo refinement:")
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

    print(f"\n  [4] Per-type bsolo refinement:")
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

    # [5] Output final results
    val_preds, val_labels = a_v2_predict(train_samples, bsolo_th, lsolo_th, mv_th, sp_th)
    test_preds, test_labels = a_v2_predict(test_samples, bsolo_th, lsolo_th, mv_th, sp_th)
    _, _, val_f1, val_tp, val_pred_n, val_true_n = evaluate(val_preds, val_labels)
    _, _, test_f1, test_tp, test_pred_n, test_true_n = evaluate(test_preds, test_labels)
    print(f"\n  [5] Final results on the validation set:")
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
# 5d. Strategy A_cal: Calibrated Weighted Fusion (calibration + per-type α weighting)
# ====================================================================
def _fit_isotonic_calibrators(samples):
    """Fit isotonic calibrators for BERT / LLM conf on `samples`.
    Input: samples (list of {'candidates': [...], 'label_set': set(...)})
    Returns: (bert_cal, llm_cal, n_bert, n_llm, n_bert_pos, n_llm_pos)
    Calibrators: cal.predict([x]) → P(correct|x), 0 ≤ y ≤ 1
    Guard: if all samples share the same label (positive or negative rate
        is 100% or 0%), sklearn IsotonicRegression will raise
        "y must be at least two classes"; in that case cal.f_ remains None,
        and the upper layer's predict falls back to a fast-path without
        calibration, so the pipeline won't crash.
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
    """In-place override of cand['bert_conf'] / cand['llm_conf'] with the
    calibrated P(correct).
    Only items with present=True are calibrated; non-present conf stays 0.0.
    Important: this function overwrites the original conf; the caller must
    deep copy first.
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
    """Calibration + per-type α weighted fusion (no consensus bonus, no factor).
    Formula: score = α(type) * cal_bert + (1-α) * cal_llm; keep if score >= th
    `samples` must be already calibrated (cand['bert_conf'/'llm_conf'] already
    represent P(correct)).
    type_alphas: dict{type: float} (per-type) or float (global α).
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
    """Fast version of `_cal_fusion_predict` returning (tps, pns, tns) np.array.
    Purpose: speed up the ~66-iteration grid search (aligned with
    `static_fusion_predict_metrics`)."""
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
    """Strategy A_cal: Calibrated Weighted Fusion (calibration + per-type α weighting).
    Steps:
      1) Fit isotonic calibrators on train_samples
      2) Deep-copy train/test samples and apply calibration in-place
      3) Global grid search (α, th) → per-type α refinement
      4) Output val / test P/R/F1
    Returns: (val_f1, test_f1, type_alphas, threshold, train_cal, test_cal, cal_info)
    """
    import copy as _copy
    print("\n" + "=" * 70)
    print(f"【Strategy {label}】Calibrated Weighted Fusion (calibration + per-type α weighting)")
    print("=" * 70)
    print("Design rationale:")
    print("  1) Isotonic regression calibrates BERT/LLM conf → P(correct|conf)")
    print("     to correct the bias caused by LLM conf being too high / concentrated")
    print("  2) Per-type α weighting:  score = α*cal_bert + (1-α)*cal_llm")
    print("  3) Grid search (α ∈ [0,1], th ∈ [0.2,0.7]) → per-type α refinement")
    print("  4) Single threshold th (no consensus bonus, no bert/llm_only factor)")
    print("-" * 70)

    # 1) Fit calibrators
    print("\n  [1] Fit isotonic calibrators (on train_samples) ...")
    bert_cal, llm_cal, n_b, n_l, n_b_pos, n_l_pos = _fit_isotonic_calibrators(train_samples)
    bert_rate = n_b_pos / n_b if n_b > 0 else 0.0
    llm_rate = n_l_pos / n_l if n_l > 0 else 0.0
    print(f"      BERT: {n_b} samples, positive rate = {bert_rate:.3f}, "
          f"calibrator fitted={bert_cal.f_ is not None}")
    print(f"      LLM:  {n_l} samples, positive rate = {llm_rate:.3f}, "
          f"calibrator fitted={llm_cal.f_ is not None}")
    cal_info = {
        'n_bert': n_b, 'n_llm': n_l,
        'bert_pos_rate': bert_rate, 'llm_pos_rate': llm_rate,
        'bert_cal_fitted': bert_cal.f_ is not None,
        'llm_cal_fitted': llm_cal.f_ is not None,
    }

    # 2) Deep copy + apply calibration
    print("\n  [2] Apply calibration (deep copy to avoid polluting the original samples) ...")
    train_cal = _copy.deepcopy(train_samples)
    test_cal = _copy.deepcopy(test_samples)
    _calibrate_samples_inplace(train_cal, bert_cal, llm_cal)
    _calibrate_samples_inplace(test_cal, bert_cal, llm_cal)

    # 3) Global grid search (α, th) — calibrated conf ∈ [0,1], threshold range matches v3
    print("\n  [3] Global grid search (α ∈ [0, 1] step 0.1, th ∈ [0.2, 0.7] step 0.05) ...")
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
                print(f"      [{cnt}/{total}] elapsed {time.time()-t0:.1f}s, "
                      f"current best F1={best_f1:.4f}")
    print(f"  -> Global best: α={best_alpha:.2f}, th={best_th:.2f}, F1={best_f1:.4f}")

    # 4) Per-type α refinement
    print(f"\n  [4] Per-type α refinement (th fixed = {best_th:.2f}, α ∈ [0,1] step 0.05) ...")
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

    # 5) Final on validation set
    print(f"\n  [5] Final results on the validation set (α and threshold above):")
    val_preds, val_labels = _cal_fusion_predict(train_cal, type_alphas, best_th)
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels, tol=POSITION_TOLERANCE)
    print_metrics(f"{label} (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    # 6) Test set
    test_preds, test_labels = _cal_fusion_predict(test_cal, type_alphas, best_th)
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels, tol=POSITION_TOLERANCE)
    print_metrics(f"{label} (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)

    # 7) Calibration contribution ablation — turn calibration off, see how much
    #    F1 a pure per-type α weighting can deliver
    print(f"\n  [7] Calibration contribution ablation (same search on uncalibrated conf; "
          f"ΔF1 = calibration contribution):")
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
    print(f"      Calibrated   test F1 = {test_f1:.4f}  (α per-type, th={best_th:.2f})")
    print(f"      Uncalibrated test F1 = {unc_test_f1:.4f}  (α per-type, th={best_unc_th:.2f})")
    print(f"      Calibration contribution ΔF1 = {cal_delta:+.4f}")

    return val_f1, test_f1, type_alphas, best_th, train_cal, test_cal, cal_info


def _predict_with_calibrators_reuse_alphas(target_samples, train_for_fit,
                                            type_alphas, threshold):
    """Refit the calibrators on `train_for_fit`, recalibrate `target_samples`,
    and produce predictions directly with the already-searched
    `type_alphas` / `threshold`.
    Purpose: in the strict evaluation block, A_cal's val evaluation cannot
        use train_cal (in-sample). It must use va_samples (8:2 split, 800
        items) that are unseen by the gating network.
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
    """drop_beam_features=True: train the gating network without the 5-beam
    specific features (used for the B_b1 ablation).
    Two network architectures:
      - 5b mode (drop_beam=False): 2×hidden=64 + 3 heads (w_bert/w_llm/bonus)
        fusion = w_bert*bert_conf + w_llm*llm_conf + bonus*both_pres*0.30
        loss: BCE(combined, label) + 0.3 * sample-level ranking
      - b1 mode (drop_beam=True):  1×hidden=32 + 1 head, single logit
        fusion = sigmoid(logit)
        loss: BCE(logit, label) + 0.15 * MSE(sigmoid(logit), 0.5*llm+0.5*bert)
        (the concrete switch is handled inside `train_gating` by checking
         `model_tag=='b1'`)
    """
    # Pass model_tag to train_gating so 5b / b1 are saved to distinct file names
    model_tag = 'b1' if drop_beam_features else '5b'
    print("\n" + "=" * 70)
    print("【Strategy B】Gating Network Fusion")
    print("=" * 70)
    if drop_beam_features:
        print("Mode: b1 (Beam-1 only) — single 32-dim head, sigmoid(logit) → score")
        print("      Loss: BCE + 0.15 * MSE(sigmoid(logit), 0.5*llm+0.5*bert)")
    else:
        print("Mode: 5b — 3 heads (w_bert, w_llm, bonus)")
        print("      Fusion = w_bert*bert_conf + w_llm*llm_conf + bonus*both_pres*0.30")
        print("      Loss: BCE(combined, label) + 0.3 * sample-level ranking")
    print("-" * 70)

    print("  [1] Train/validation split (8:2) ...")
    n = len(train_samples)
    rng = np.random.RandomState(GLOBAL_SEED)
    perm = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx = set(perm[:n_val].tolist())
    tr_samples = [s for i, s in enumerate(train_samples) if i not in val_idx]
    va_samples = [s for i, s in enumerate(train_samples) if i in val_idx]

    feats_tr, tids_tr, gids_tr, labs_tr = _candidates_to_features(tr_samples, drop_beam_features=drop_beam_features)
    feats_va, tids_va, gids_va, labs_va = _candidates_to_features(va_samples, drop_beam_features=drop_beam_features)
    print(f"      Train candidates: {len(feats_tr)}  Validation candidates: {len(feats_va)}")
    train_ds = TensorDataset(*_to_tensors(feats_tr, tids_tr, gids_tr, labs_tr,
                                            drop_beam_features=drop_beam_features))
    valid_ds = TensorDataset(*_to_tensors(feats_va, tids_va, gids_va, labs_va,
                                            drop_beam_features=drop_beam_features))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False)

    print(f"\n  [2] Train gating network (epochs={epochs}, lr={lr}, drop_beam={drop_beam_features}) ...")
    n_feats = 12 if drop_beam_features else 16
    model = GatingNetwork(num_types=NUM_TYPES, n_feats=n_feats)
    t0 = time.time()
    model = train_gating(model, train_loader, valid_loader,
                         epochs=epochs, lr=lr, patience=5, save_dir=save_dir,
                         model_tag=model_tag)
    print(f"      Training time: {time.time() - t0:.1f}s")

    print("\n  [3] Global threshold grid search (on validation set, with P/R balance):")
    # Speed-up: one batched inference retrieves every sample's scores; subsequent
    # th sweeps reuse them (avoiding ~40×800 model forwards)
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
    # Also find the P≈R threshold (smallest gap), as a fallback — reuse
    # val_scores, no re-inference needed
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
    print(f"  -> Best F1 globally: th={best_th:.2f}, F1={best_f1:.4f}")
    print(f"  -> Best P/R balance: th={best_balanced_th:.2f}  (P-R gap={best_balanced_gap:.4f})")
    # If the balanced-threshold F1 is not significantly worse, prefer balance
    if best_balanced_f1 >= best_f1 - 0.003:
        print(f"  -> Adopting P/R-balanced threshold {best_balanced_th} "
              f"(F1={best_balanced_f1:.4f}, balance is better)")
        best_th = best_balanced_th

    # Note: the gating network's combined score is already a per-sample continuous
    # value, so a global threshold is sufficient.
    print("\n  [4] Per-type refinement (range -0.10 ~ +0.10, anti-overfit):")
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

    # Final results on validation / test set: reuse val_scores, recompute
    # test_scores (one-shot)
    print("\n  [5] Final results on the validation set:")
    val_preds, val_labels = _filter_by_threshold(va_samples, val_scores, best_type_th)
    val_p, val_r, val_f1, val_tp, vp, vt = evaluate(val_preds, val_labels)
    print_metrics("Gating-Network (val)", val_p, val_r, val_f1, val_tp, vp, vt)

    test_scores = _gating_batch_scores(test_samples, model, drop=getattr(model, 'drop_beam_features', False))
    test_preds, test_labels = _filter_by_threshold(test_samples, test_scores, best_type_th)
    test_p, test_r, test_f1, test_tp, tp_n, tt_n = evaluate(test_preds, test_labels)
    print_metrics("Gating-Network (test)", test_p, test_r, test_f1, test_tp, tp_n, tt_n)
    # Returns (val_f1, test_f1, type_th, (val_p/r, test_p/r), model) — `model`
    # is reused for the strict paper-side evaluation
    return val_f1, test_f1, best_type_th, (val_p, val_r, test_p, test_r), model, va_samples


# ====================================================================
# 7. Main pipeline
# ====================================================================
def main():
    # ---------- Configuration ----------
    # Choose the dataset here (matches the selectable options in the original main)
    LLM_FILE = "evl_f1/mark/glm4_ner_confidence_beams5.jsonl"
    #LLM_FILE = "evl_f1/mark/lla_ner_Llama3-8B-confidence_beams_5.jsonl"
    BERT_FILE = "evl_f1/EGP_chinese-roberta-wwm-ext.jsonl"
    BERT_FILE = "evl_f1/TP_chinese-roberta-wwm-ext.jsonl"

    LLM_SOURCE = 'ner_models'
    USE_BEAM = True  # 5-beam voting

    TRAIN_SAMPLE_SIZE = 4000  # Training (incl. validation) sample count
    TEST_SAMPLE_SIZE = 1000   # Test sample count
    EPOCHS = 30               # Gating network training epochs
    LR = 2e-3
    SAVE_DIR = "saved_models_clean"

    print("=" * 70)
    print("LLM + BERT Entity Fusion (clean rewrite)")
    print("=" * 70)
    print(f"  LLM file:  {LLM_FILE}")
    print(f"  BERT file: {BERT_FILE}")
    print(f"  LLM data source: {LLM_SOURCE}")
    print(f"  Beam voting:     {USE_BEAM}")
    print(f"  Position tolerance: {POSITION_TOLERANCE}")
    print()

    # 1) Load + split
    print("[Step 1] Load data and split train/test ...")
    llm_lines, bert_lines = load_data(LLM_FILE, BERT_FILE)
    print(f"  Loaded: {len(llm_lines)} entries (min of the two)")
    (train_llm, train_bert), (test_llm, test_bert) = split_train_test(
        llm_lines, bert_lines, test_size=TEST_SAMPLE_SIZE, seed=GLOBAL_SEED)
    # Control training sample count
    train_llm = train_llm[:TRAIN_SAMPLE_SIZE]
    train_bert = train_bert[:TRAIN_SAMPLE_SIZE]
    print(f"  Training pool (for 8:2 split in gating): {len(train_llm)} entries")
    print(f"  Test:                                   {len(test_llm)} entries")
    # Note: the validation set is split 8:2 inside `run_gating_fusion`
    # (n=tr//5); not split here in main.

    # 2) Parse → samples (use α=0.7, β=0.05 as placeholders first; the optimal
    #    values from [Step 1.5] are applied later via re-aggregation)
    print("\n[Step 2] Parse training set (initial, α=0.7 β=0.05 placeholders) ...")
    global LLM_ALPHA, LLM_VOTE_REWARD
    LLM_ALPHA = 0.7
    LLM_VOTE_REWARD = 0.05
    train_samples = build_samples(train_llm, train_bert, source=LLM_SOURCE,
                                  tol=POSITION_TOLERANCE, use_beam=USE_BEAM)
    print(f"  Training samples (valid): {len(train_samples)}")
    print("\n[Step 3] Parse test set (initial, α=0.7 β=0.05 placeholders) ...")
    test_samples = build_samples(test_llm, test_bert, source=LLM_SOURCE,
                                 tol=POSITION_TOLERANCE, use_beam=USE_BEAM)
    print(f"  Test samples (valid): {len(test_samples)}")

    # 3.1) [Beam-1 only copy] for ablation experiments: simulates the fair
    #      control where the LLM service returns only 1 candidate
    #      `parse_llm_line`'s `use_beam=False` branch has been changed to
    #      `_parse_mark_format_beam(models[:1], ...)`, so cand['llm_conf'] is
    #      the true Beam-1 conf, vote_count=1, and there is no 5-beam vote signal
    print("\n[Step 3.1] Parse Beam-1 only copy (for ablation: A_b1 / B_b1) ...")
    train_samples_b1 = build_samples(train_llm, train_bert, source=LLM_SOURCE,
                                      tol=POSITION_TOLERANCE, use_beam=False)
    test_samples_b1 = build_samples(test_llm, test_bert, source=LLM_SOURCE,
                                     tol=POSITION_TOLERANCE, use_beam=False)
    print(f"  Beam-1 train samples: {len(train_samples_b1)}  test samples: {len(test_samples_b1)}")

    # 1.5) Search optimal (α, β) (max+mean+vote_reward), using a simplified
    # A strategy's train F1 as the target.
    # The A strategy is sensitive to llm_conf changes, making it a good
    # aggregation evaluator compared with A''.
    # Note: (α, β) is searched on train_samples (3000); the strict evaluation
    # block's val column is computed on the 800 items unseen by the gating network
    if USE_BEAM:
        print("\n[Step 1.5] Search (α, β) parameters (max+mean+vote_reward); "
              "objective is A strategy's F1 on train_samples ...")
        # Speed-up: only re-aggregate conf (via _reaggregate_llm_confs) in the
        # 30 grid evaluations, do NOT rebuild_samples
        import copy
        train_samples_base = copy.deepcopy(train_samples)  # original llm_conf as template
        # Simplified A strategy (α_per_type = F1_bert / (F1_bert + F1_llm),
        # bonus = 0.05, th = 0.5)
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
        # β range extended to [-0.05, +0.10] including 0, to confirm that the
        # vote_reward soft bonus truly carries no signal for the A strategy
        for alpha_try in [1.00, 0.85, 0.70, 0.55, 0.40]:
            for beta_try in [-0.05, 0.00, 0.03, 0.05, 0.08, 0.10]:
                # Deep copy from base each time, recompute conf (in-place; base is unchanged)
                _train = copy.deepcopy(train_samples_base)
                _reaggregate_llm_confs(_train, alpha_try, beta_try)
                _f1 = _quick_a_f1(_train)
                marker = " *" if _f1 > best_alphabeta_f1 else ""
                print(f"      α={alpha_try:.2f}  β={beta_try:.2f}  A val F1={_f1:.4f}{marker}")
                if _f1 > best_alphabeta_f1:
                    best_alphabeta_f1, best_alpha, best_beta = _f1, alpha_try, beta_try
        print(f"  -> Best α={best_alpha:.2f}, β={best_beta:.2f}  (A val F1 = {best_alphabeta_f1:.4f})")
        LLM_ALPHA = best_alpha
        LLM_VOTE_REWARD = best_beta
        # Re-aggregate llm_conf for train/test with the optimal parameters
        # (no need to rebuild_samples)
        print(f"\n  [Recompute conf] Use α={best_alpha}, β={best_beta} to re-aggregate LLM conf ...")
        _reaggregate_llm_confs(train_samples, best_alpha, best_beta)
        _reaggregate_llm_confs(test_samples,  best_alpha, best_beta)

    # 2b) Print beam vote distribution
    if USE_BEAM:
        from collections import Counter
        vote_dist = Counter()
        for s in train_samples + test_samples:
            for c in s['candidates']:
                if c['llm_present']:
                    vote_dist[int(c.get('vote_count', 1))] += 1
        print("\n  [Beam vote distribution] (vote counts for LLM-only candidates):")
        for v in sorted(vote_dist):
            print(f"    {v} votes: {vote_dist[v]} candidates")
        # Average vote count of consensus (BERT+LLM) candidates
        both_votes = [c.get('vote_count', 1) for s in train_samples + test_samples
                      for c in s['candidates'] if c['bert_present'] and c['llm_present']]
        if both_votes:
            print(f"    Average vote count of consensus candidates: {sum(both_votes) / len(both_votes):.2f} "
                  f"(total {len(both_votes)})")
        solo_votes = [c.get('vote_count', 1) for s in train_samples + test_samples
                      for c in s['candidates']
                      if c['llm_present'] and not c['bert_present']]
        if solo_votes:
            print(f"    Average vote count of LLM-only candidates: {sum(solo_votes) / len(solo_votes):.2f} "
                  f"(total {len(solo_votes)})")

    # 3) Baseline metrics
    print("\n" + "=" * 70)
    print("【Baseline】Single model")
    print("=" * 70)
    bert_preds, bert_labels = bert_only_predict(train_samples, train_llm, train_bert, source=LLM_SOURCE)
    bp_v, br_v, bf1_v, btp_v, bpn_v, btn_v = evaluate(bert_preds, bert_labels)
    print_metrics("BERT Only (val)", bp_v, br_v, bf1_v, btp_v, bpn_v, btn_v)
    bert_test_preds, bert_test_labels = bert_only_predict(test_samples, test_llm, test_bert, source=LLM_SOURCE)
    bp, br, bf1_t, btp, bpn, btn = evaluate(bert_test_preds, bert_test_labels)
    print_metrics("BERT Only (test)", bp, br, bf1_t, btp, bpn, btn)

    # 3) LLM-Only multiple baselines (Top-1 ~ Top-5 + 5-union; paper's strict baseline = Top-1)
    print("\n" + "=" * 90)
    print("【LLM-Only Baselines · Strict Split】 (LLM service returns only 1 best → paper's main baseline = Top-1)")
    print("=" * 90)
    print(f"  {'Baseline':<24} | {'val (train)':^30} | {'test':^30}")
    print(f"  {'':<24} | {'P':>7} {'R':>7} {'F1':>7} | {'P':>7} {'R':>7} {'F1':>7}")
    print("  " + "-" * 78)
    # Top-1 (paper's main baseline) ~ Top-5
    llm_baselines = {}
    for k in range(5):
        v_preds, v_labels = llm_only_predict(train_samples, beam_idx=k)
        p, r, f1, _, _, _ = evaluate(v_preds, v_labels)
        t_preds, t_labels = llm_only_predict(test_samples, beam_idx=k)
        tp, tr, tf1, _, _, _ = evaluate(t_preds, t_labels)
        llm_baselines[k] = (p, r, f1, tp, tr, tf1)
        print(f"  LLM Top-{k+1} (Beam {k+1})".ljust(26) +
              f" | {p:>7.4f} {r:>7.4f} {f1:>7.4f} | {tp:>7.4f} {tr:>7.4f} {tf1:>7.4f}")
    # 5-union without threshold (legacy behavior, not strict)
    v_preds, v_labels = llm_only_predict(train_samples, beam_idx=None)
    p, r, f1, _, _, _ = evaluate(v_preds, v_labels)
    t_preds, t_labels = llm_only_predict(test_samples, beam_idx=None)
    tp, tr, tf1, _, _, _ = evaluate(t_preds, t_labels)
    print(f"  LLM 5-Union (no-thr)  ".ljust(26) +
          f" | {p:>7.4f} {r:>7.4f} {f1:>7.4f} | {tp:>7.4f} {tr:>7.4f} {tf1:>7.4f}"
          f"  ← current legacy 'LLM Only' baseline")
    llm_baselines['union'] = (p, r, f1, tp, tr, tf1)
    # 5-union + conf threshold (Top-1 style, strict)
    v_preds, v_labels = llm_only_predict(train_samples, beam_idx=None, conf_th=0.5)
    p, r, f1, _, _, _ = evaluate(v_preds, v_labels)
    t_preds, t_labels = llm_only_predict(test_samples, beam_idx=None, conf_th=0.5)
    tp, tr, tf1, _, _, _ = evaluate(t_preds, t_labels)
    print(f"  LLM 5-Union (conf≥0.5)".ljust(26) +
          f" | {p:>7.4f} {r:>7.4f} {f1:>7.4f} | {tp:>7.4f} {tr:>7.4f} {tf1:>7.4f}"
          f"  ← conf-filtered")
    print("  " + "-" * 78)
    # Paper's main baseline: Top-1
    lp_v, lr_v, lf1_v, lp, lr, lf1_t = llm_baselines[0]

    # 4) Strategy A: Static Weight Fusion (3-case soft fusion)
    a_val, a_test, a_alphas, a_th, a_hyper = run_static_fusion(train_samples, test_samples)
    # Recompute val/test P/R for the summary (must use the hyperparameters
    # searched inside `run` via `a_hyper`; hardcoding would be inconsistent
    # with the internal run)
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

    # 4b) Strategy A'': Vote-Aware Hard-Rule
    app_val, app_test, app_th, app_mvb, app_mvl = run_static_fusion_voteaware(
        train_samples, test_samples)
    app_vp, app_vr, _, _, _, _ = evaluate(
        static_fusion_voteaware_predict(train_samples, app_th, app_mvb, app_mvl)[0],
        [s['label_entities'] for s in train_samples])
    app_tp, app_tr, _, _, _, _ = evaluate(
        static_fusion_voteaware_predict(test_samples, app_th, app_mvb, app_mvl)[0],
        [s['label_entities'] for s in test_samples])

    # 4c) Strategy A_v2: Consensus V2
    av2_val, av2_test, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d, av2_stats = run_a_v2(
        train_samples, test_samples)
    av2_vp, av2_vr, _, _, _, _ = evaluate(
        a_v2_predict(train_samples, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d)[0],
        [s['label_entities'] for s in train_samples])
    av2_tp, av2_tr, _, _, _, _ = evaluate(
        a_v2_predict(test_samples, av2_bsolo, av2_lsolo, av2_mv_d, av2_ms_d)[0],
        [s['label_entities'] for s in test_samples])

    # 4d) Strategy A_cal: Calibrated Weighted Fusion (calibration + per-type α weighting)
    #     From gating_network_dp_mark_v3.py strategies 3/9; fit calibrators on
    #     train_samples, and run pure 2-feature weighting (no consensus bonus)
    #     on deep-copied train_cal / test_cal.
    #     Note: under 5-beam mode c['llm_conf'] is already aggregated as
    #         α·max + (1-α)·mean + β·(N-1); calibration can still learn
    #         "is high conf actually correct" → reduces LLM-dominance bias.
    acal_val, acal_test, acal_alphas, acal_th, train_cal, test_cal, acal_info = \
        run_calibrated_weighted_fusion(train_samples, test_samples, label='A_cal')
    # Recompute val/test P/R for the summary (must use the hyperparameters
    # searched inside `run` via `acal_alphas` / `acal_th`)
    acal_v_preds, _ = _cal_fusion_predict(train_cal, acal_alphas, acal_th)
    acal_val_p, acal_val_r, _, _, _, _ = evaluate(
        acal_v_preds, [s['label_entities'] for s in train_cal], tol=POSITION_TOLERANCE)
    acal_t_preds, _ = _cal_fusion_predict(test_cal, acal_alphas, acal_th)
    acal_test_p, acal_test_r, _, _, _, _ = evaluate(
        acal_t_preds, [s['label_entities'] for s in test_cal], tol=POSITION_TOLERANCE)

    # 5) Strategy B: Gating Network Fusion
    b_val, b_test, b_th, (b_vp, b_vr, b_tp, b_tr), gating_model, va_samples = run_gating_fusion(
        train_samples, test_samples,
        train_llm, train_bert,
        source=LLM_SOURCE, epochs=EPOCHS, lr=LR, save_dir=SAVE_DIR,
    )

    # 5.5) Per-beam P/R/F1 of the LLM (5 beams) for comparison
    print("\n" + "=" * 90)
    print("【LLM 5-Beam Per-Beam Performance】 (by beam index, 0=beam1 highest accuracy)")
    print("=" * 90)
    beam_results = []
    for k in range(5):
        bv_p, bv_r, bv_f1, _, _, _ = eval_llm_beam_k(train_llm, train_bert, beam_idx=k, source=LLM_SOURCE)
        bt_p, bt_r, bt_f1, _, _, _ = eval_llm_beam_k(test_llm,  test_bert,  beam_idx=k, source=LLM_SOURCE)
        beam_results.append((bv_p, bv_r, bv_f1, bt_p, bt_r, bt_f1))
        print(f"  LLM Beam-{k+1} (val)  P={bv_p:.4f}  R={bv_r:.4f}  F1={bv_f1:.4f}")
        print(f"  LLM Beam-{k+1} (test) P={bt_p:.4f}  R={bt_r:.4f}  F1={bt_f1:.4f}")

    # 5.6) 5-beam vote (5-way union) vs single-beam comparison
    print("\n【5-beam union (no threshold)】")
    union_preds_v, union_labels_v = [], []
    union_preds_t, union_labels_t = [], []

    def _union_pred_and_llm_label(ll, bl):
        """5-beam union preds + LLM file's `label` field (mark format, human-annotated)"""
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
        # Ground truth uses the LLM file's `label` field (human annotation),
        # not the BERT file (BERT contains inference results)
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
    # 5.7) [Ablation A_b1 / B_b1] Build samples from Beam-1 only
    #     - cand['llm_conf'] = Beam-1 true conf (no 5-beam aggregation)
    #     - cand['vote_count'] = 1 (no multi-beam vote signal)
    #     Purpose: contrast "5-beam voting" vs "single beam" to quantify
    #     the contribution of voting to fusion
    # ============================================================
    print("\n" + "=" * 90)
    print("【Ablation A_b1 / B_b1】 Using Beam-1 only (cand.llm_conf = Beam-1 true conf, vote_count=1)")
    print("=" * 90)

    # 5.7a) A_b1: Static Weight Fusion (Beam-1 only)
    print("\n  [A_b1] Static Weight Fusion · Beam-1 only")
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

    # 5.7b) B_b1: Gating Network (Beam-1 only)
    # Important: under Beam-1 mode, cand['vote_count']=1, cand['llm_max_conf']
    #   = cand['llm_avg_conf'] = cand['llm_conf'], cand['best_beam_idx']=0 — these
    #   4 "5-beam specific" features all degenerate to constants. Feeding them
    #   would make the model learn "vote=1=untrustworthy" and collapse.
    # Fix: use drop_beam_features=True to keep the gating network from seeing
    #      those 4 features, forcing it to use only the 12 generic ones.
    # And: to align with gating_network_dp_mark_v3.py's Gating Network behavior,
    #      use more stable hyperparameters (epochs=100, lr=5e-4, patience=15)
    #      + BCE + 0.15*MSE distillation loss (the switch is handled inside
    #      train_gating by checking model_tag=='b1')
    print("\n  [B_b1] Gating Network · Beam-1 only (drop_beam_features=True)")
    bb1_val, bb1_test, bb1_th, (bb1_vp, bb1_vr, bb1_tp, bb1_tr), bb1_model, bb1_va = run_gating_fusion(
        train_samples_b1, test_samples_b1,
        train_llm, train_bert,
        source=LLM_SOURCE, epochs=100, lr=5e-4, save_dir=SAVE_DIR,
        drop_beam_features=True,
    )
    # bb1_va is the 8:2 split validation set on the b1 samples (split inside
    # run_gating_fusion with GLOBAL_SEED). A_b1's strict evaluation reuses the
    # same split, so both b1 strategies' val/test come from the same partition
    va_samples_b1 = bb1_va

    # 5.7c) A_cal_b1: Calibrated Weighted Fusion (Beam-1 only)
    #     Refit the calibrators on train_samples_b1 — the 5-beam calibrators
    #     cannot be reused, because the LLM conf distribution (single beam vs
    #     5-beam aggregated) is different → calibration scales mismatch.
    #     This ablation answers the question "can v3 log +0.0033 be reproduced
    #     in st8 under Beam-1 mode"
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
    # The strict evaluation block for b1 also uses the same split (8:2), aligned with B_b1

    # 6) Summary (P / R / F1 columns)
    # Note: the val column of this table is the F1 on train_samples (3000),
    # which contains the gating network's training samples (data leakage).
    # The truly-unseen val performance is in the "【Paper Strict Evaluation】"
    # block below (val=va_samples, 800 items, unseen by the gating network)
    print("\n" + "=" * 90)
    print("Final Summary (P / R / F1) — val column=train_samples(3000); truly-unseen val in strict block")
    print("=" * 90)
    hdr = (f"  {'Strategy':<36} | {'val (=train)':^20} | {'test':^20} | {'Δ test':>8}")
    print(hdr)
    sub = (f"  {'':<36} | {'P':>6} {'R':>6} {'F1':>6} | "
           f"{'P':>6} {'R':>6} {'F1':>6} | {'F1':>8}")
    print(sub)
    print("  " + "-" * 86)
    def row(name, vp, vr, vf, tp, tr, tf):
        delta = tf - bf1_t
        return (f"  {name:<36} | {vp:>6.4f} {vr:>6.4f} {vf:>6.4f} | "
                f"{tp:>6.4f} {tr:>6.4f} {tf:>6.4f} | {delta:>+8.4f}")
    print(row("[Baseline] BERT Only", bp_v, br_v, bf1_v, bp, br, bf1_t))
    print(row("[Baseline] LLM Only",  lp_v, lr_v, lf1_v, lp, lr, lf1_t))
    # 5 LLM beam individual baselines (per user note: beam 1 highest accuracy, beam 5 lowest)
    for k in range(5):
        bvp, bvr, bvf, btp, btr, btf = beam_results[k]
        print(row(f"[Baseline] LLM Beam-{k+1}", bvp, bvr, bvf, btp, btr, btf))
    print(row("[Baseline] LLM 5-beam union (no threshold)", uvp, uvr, uvf1, utp, utr, utf1))
    print(row("A. Static Weight", a_val_p, a_val_r, a_val, a_test_p, a_test_r, a_test))
    print(row("A''. Vote-Aware Hard-Rule",     app_vp, app_vr, app_val, app_tp, app_tr, app_test))
    print(row("A_v2. Consensus",               av2_vp, av2_vr, av2_val, av2_tp, av2_tr, av2_test))
    print(row("A_cal. Calibrated Weighted",    acal_val_p, acal_val_r, acal_val,
              acal_test_p, acal_test_r, acal_test))
    print(f"  [Ablation] A_cal calibration contribution: calibrated={acal_test:.4f} "
          f"vs uncalibrated={acal_info.get('unc_test_f1', 0):.4f}"
          f" → Δ={acal_test - acal_info.get('unc_test_f1', 0):+.4f}")
    print(row("B. Gating Network",             b_vp, b_vr, b_val, b_tp, b_tr, b_test))
    print("  --- Ablation: Beam-1 only (cand.llm_conf = Beam-1 true conf, no 5-beam voting) ---")
    # Note: the val column for the b1 strategies is in-sample (val=train_samples_b1,
    # contains training samples, data leakage). The truly-unseen val (bb1_va)
    # performance is in the "【Paper Strict Evaluation】" block, rows A_b1/B_b1 —
    # same names so the two tables are easy to compare
    print(row("A_b1. Static (Beam-1)",     ab1_vp, ab1_vr, ab1_vf1, ab1_tp, ab1_tr, ab1_tf1))
    print(row("A_cal_b1. Calibrated (Beam-1)", acalb1_vp, acalb1_vr, acalb1_vf1,
              acalb1_tp, acalb1_tr, acalb1_tf1))
    print(f"  [Ablation] A_cal_b1 calibration contribution: calibrated={acalb1_tf1:.4f} "
          f"vs uncalibrated={acalb1_info.get('unc_test_f1', 0):.4f}"
          f" → Δ={acalb1_tf1 - acalb1_info.get('unc_test_f1', 0):+.4f}")
    print(row("B_b1. Gating (Beam-1)",     bb1_vp, bb1_vr, bb1_val, bb1_tp, bb1_tr, bb1_test))
    print("=" * 90)

    # ============================================================
    # 7) [Paper Strict Evaluation] Re-evaluate with tol=POSITION_TOLERANCE
    #    + Bootstrap 95% CI + significance test
    # ============================================================
    print("\n" + "=" * 90)
    print("【Paper Strict Evaluation】 tol=POSITION_TOLERANCE (aligned with build_candidates) "
          "+ Bootstrap 95% CI")
    print("=" * 90)
    TOL_EVAL = POSITION_TOLERANCE  # Paper's main evaluation = 2 (aligned with candidate generation)

    def _label_list(samples):
        return [s['label_entities'] for s in samples]

    # Recompute preds for all strategies and collect them into a dict
    # val column uses va_samples (the 800 items split 8:2 inside run_gating_fusion;
    # truly unseen by the gating network)
    # test column uses test_samples (1000 items, final test)
    # Do NOT use train_samples (data leakage: the gating network was trained on tr_samples)
    strategy_preds = {}
    # 1) BERT Only
    bp_v_pre, _ = bert_only_predict(va_samples, train_llm, train_bert, source=LLM_SOURCE)
    bp_t_pre, _ = bert_only_predict(test_samples, test_llm, test_bert, source=LLM_SOURCE)
    strategy_preds['BERT Only']        = (bp_v_pre, bp_t_pre)
    # 2) LLM Only (Top-1, paper's main baseline)
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
    # Important: the val strict evaluation uses va_samples (8:2 split, unseen
    # by the gating network) + 5-beam calibrators; test uses test_samples +
    # 5-beam calibrators. `acal_alphas/th` were searched inside
    # `run_calibrated_weighted_fusion` on train_samples; the same alphas/th
    # are applied on va_samples (unseen, 800 items) to recompute scores →
    # strictly no data leakage.
    # Note: acal_train_cal / acal_test_cal are deep-copied calibrated samples
    # and cannot be used directly on va_samples (would skip the calibration
    # step). Solution: recalibrate va_samples / test_samples inside the strict
    # block.
    av_cal_pre = _predict_with_calibrators_reuse_alphas(
        va_samples, train_samples, acal_alphas, acal_th)
    at_cal_pre = _predict_with_calibrators_reuse_alphas(
        test_samples, train_samples, acal_alphas, acal_th)
    strategy_preds['A_cal. Calibrated']  = (av_cal_pre, at_cal_pre)
    # 5b') A_cal_b1: Calibrated Weighted Fusion (Beam-1 only)
    # Important: the strict evaluation must use b1-parsed samples
    # (va_samples_b1 / test_samples_b1), not 5b-parsed va_samples / test_samples
    # — the 5b llm_conf is already aggregated as
    # α·max+(1-α)·mean+β·(N-1), which has a different distribution from the
    # b1 single-beam conf. Applying acalb1_alphas/th to the wrong distribution
    # makes F1 collapse badly.
    av_calb1_pre = _predict_with_calibrators_reuse_alphas(
        va_samples_b1, train_samples_b1, acalb1_alphas, acalb1_th)
    at_calb1_pre = _predict_with_calibrators_reuse_alphas(
        test_samples_b1, train_samples_b1, acalb1_alphas, acalb1_th)
    strategy_preds['A_cal_b1. Calibrated (Beam-1)'] = (av_calb1_pre, at_calb1_pre)
    # 6) A_b1 (Ablation: Static Weight + Beam-1 only)
    # Important: the strict evaluation must use b1-parsed samples
    # (va_samples_b1 / test_samples_b1), not 5b-parsed va_samples /
    # test_samples — the 5b llm_conf is already aggregated as
    # α·max+(1-α)·mean+β·(N-1), which has a different distribution from the
    # b1 single-beam conf. Applying ab1_alphas/ab1_th to the wrong
    # distribution makes F1 collapse badly.
    # (old val 0.7205 / test 0.7222 → after fix should approach the
    # main-table 0.7519/0.7483)
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
        # Speed-up: one-shot batched inference to get val/test scores, then
        # filter by best_type_th
        bv_scores = _gating_batch_scores(va_samples, gating_model,
                                          drop=getattr(gating_model, 'drop_beam_features', False))
        bt_scores = _gating_batch_scores(test_samples, gating_model,
                                          drop=getattr(gating_model, 'drop_beam_features', False))
        bv_pre, _ = _filter_by_threshold(va_samples, bv_scores, b_th)
        bt_pre, _ = _filter_by_threshold(test_samples, bt_scores, b_th)
        strategy_preds['B. Gating Net'] = (bv_pre, bt_pre)
    else:
        print("  [skip B] gating_model is not in main's scope, skip B's strict evaluation")
    # 7) B_b1 (Ablation: Gating + Beam-1 only)
    # Important: the strict evaluation must use b1-parsed samples
    # (bb1_va / test_samples_b1), not 5b-parsed va_samples / test_samples —
    # bb1_model was trained on 12-d generic features; applying it to 5b
    # samples' llm_conf distribution will mismatch, F1 collapses badly.
    if bb1_model is not None:
        bv_b1_scores = _gating_batch_scores(bb1_va, bb1_model,
                                             drop=getattr(bb1_model, 'drop_beam_features', False))
        bt_b1_scores = _gating_batch_scores(test_samples_b1, bb1_model,
                                             drop=getattr(bb1_model, 'drop_beam_features', False))
        bv_b1_pre, _ = _filter_by_threshold(bb1_va, bv_b1_scores, bb1_th)
        bt_b1_pre, _ = _filter_by_threshold(test_samples_b1, bt_b1_scores, bb1_th)
        strategy_preds['B_b1. Gating (Beam-1)'] = (bv_b1_pre, bt_b1_pre)
    else:
        print("  [skip B_b1] bb1_model is not in main's scope, skip B_b1's strict evaluation")

    val_labels = _label_list(va_samples)   # labels of the 800 items unseen by the gating network (no overlap with tr_samples)
    test_labels = _label_list(test_samples)

    # Speed-up: precompute (tps, pns, tns) for all strategies once, so the
    # bootstrap resampling does not redo set ops
    print(f"\n  [Precompute] 8 strategies × (val, test) = 16 metric groups, compute once ...")
    pre_v = {name: _precompute_metrics(vp, val_labels, tol=TOL_EVAL)
             for name, (vp, _) in strategy_preds.items()}
    pre_t = {name: _precompute_metrics(tp, test_labels, tol=TOL_EVAL)
             for name, (_, tp) in strategy_preds.items()}

    # Re-evaluate P/R/F1 with tol=2
    print(f"\n  tol = {TOL_EVAL} (aligned with build_candidates)")
    # Header: val/test columns of P / R / F1 + 95% CI
    print(f"  {'Strategy':<22} | {'val P':>7} {'val R':>7} {'val F1':>7} {'95% CI':<17} | "
        f"{'test P':>7} {'test R':>7} {'test F1':>7} {'95% CI':<17} | {'Δ F1':>8}")
    print("  " + "-" * 120)

    strict_results = {}
    for name, (vp, tp) in strategy_preds.items():
        tps_v, pns_v, tns_v = pre_v[name]
        tps_t, pns_t, tns_t = pre_t[name]
        # Overall P/R/F1
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

    # Paired-bootstrap significance test on key comparisons (uses precomputed arrays)
    print(f"\n  【Significance Test】 Paired Bootstrap (H0: ΔF1 = 0, n_boot=1000):")
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
