"""
LLM Named Entity Recognition (NER) Evaluation Script
====================================================
Evaluates P/R/F1 against LLM output in the mark annotation format
(etype : [entity]...).
Supported match modes: indexMatch / startIndexMatch / endIndexMatch / entityMatch.
Optionally enables position tolerance (`tolerance`) and multi-position candidates
(`keep_all_matches`).
"""

import json
import re
import ast
import numpy as np


# ====================================================================
# 1. Data loading
# ====================================================================
def load_jsonl(filepath):
    """Load a jsonl file, auto-attempting json / ast parsing and skipping
    empty or unparseable lines."""
    def parse_line(line):
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(line)
            except Exception:
                return None

    with open(filepath, 'r', encoding='utf-8') as f:
        return [obj for line in f if (obj := parse_line(line)) is not None]


# ====================================================================
# 2. Original text extraction (recovered from the prompt)
# ====================================================================
# Start marker — aligned with the fixed opening of the task's prompt template
_START_MARK = "明，与成人SARS相比，儿童[细胞下降]不明显，证明上述推测成立。\n"

# Candidate LLM chat-template end markers, ordered by observed frequency
_END_MARKS = [
    '<|eot_id|>',          # Llama2 / Qwen
    '<|end_of_text|>',     # Qwen
    '<|im_end|>',          # Qwen ChatML / GLM
    '<|end_turn|>',        # Gemma
    '<|assistant|>',       # GLM4
    '<｜Assistant｜>',      # DeepSeek (full-width)
    '<|start_header_id|>', # Llama3
    '<|/assistant|>',      # Mistral
]


def get_promptPr(prompt, start_mark, end_mark):
    """Return the substring of `prompt` that lies between `start_mark` and
    `end_mark`, or None if not found."""
    if prompt is None:
        return None
    pattern = re.escape(start_mark) + '(.*?)' + re.escape(end_mark)
    match = re.search(pattern, prompt)
    return match.group(1) if match else None


def extract_original_text(sample):
    """Extract the original source text from a sample.
    Prefers the explicit `text` field; otherwise slices the `prompt` between
    the start marker and the first matching end marker.
    """
    if 'text' in sample:
        return sample['text']
    prompt = sample.get('prompt', '')
    if not prompt:
        return ""
    for end_mark in _END_MARKS:
        result = get_promptPr(prompt, _START_MARK, end_mark)
        if result:
            return result.strip()
    return ""


# ====================================================================
# 3. Parsing the mark-style annotation
# ====================================================================
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
    """Extract (start, end, entity) tuples from one mark-format content line.
    `start`/`end` are in-line offsets (after stripping the surrounding []),
    and `entity` is already stripped.
    Two input forms are supported:
      A) Original sentence with [entity] markers → exact offsets
      B) Bare entities (split on [、,;；]) → offsets within the content
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


def parse_marked_text_with_pos(text):
    """Parse mark-format text into a list of
    (start_idx, end_idx, entity, type, line_no) tuples.
    start/end are in-line offsets (after stripping []); the caller is
    expected to realign them against the full `original_text` via
    `_correct_to_original_text`.
    """
    result = []
    for line_idx, line in enumerate(text.strip().split('\n')):
        r = _split_mark_line(line)
        if not r:
            continue
        etype, content = r
        for start, end, entity in _entities_with_pos_in_content(content):
            result.append((start, end, entity, etype, line_idx))
    return result


def parse_marked_text_entity(text):
    """Parse mark-format text into a {(entity, type): entity} dictionary."""
    result = {}
    for line in text.strip().split('\n'):
        r = _split_mark_line(line)
        if not r:
            continue
        etype, content = r
        for _, _, entity in _entities_with_pos_in_content(content):
            if entity:
                result[(entity, etype)] = entity
    return result


def _correct_to_original_text(entities, original_text, tolerance=2,
                              keep_all_matches=False):
    """Realign the in-line offsets of mark entities to their true positions
    inside `original_text`.
    Correction rules:
      1. Entity not present in original_text → drop (LLM hallucination, not
         counted in evaluation).
      2. Entity present, position already correct → keep as is.
      3. Entity present, position mismatched:
         a. Exactly one occurrence → use that occurrence.
         b. Multiple occurrences:
            - keep_all_matches=False: select the occurrence closest to the
              predicted position within `tolerance`; if none qualifies, fall
              back to the closest occurrence overall.
            - keep_all_matches=True:  keep every occurrence as a candidate.
    """
    if not original_text:
        return entities
    corrected = []
    for start_idx, end_idx, entity, etype, line_no in entities:
        if not entity:
            continue
        # 1. Drop hallucinations
        if entity not in original_text:
            continue
        # 2. Position already correct
        if 0 <= start_idx <= end_idx < len(original_text) \
                and original_text[start_idx:end_idx + 1] == entity:
            corrected.append((start_idx, end_idx, entity, etype, line_no))
            continue
        # 3. Search every occurrence
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


# ====================================================================
# 4. Evaluation
# ====================================================================
ENT_TYPES = ['bod', 'dis', 'dru', 'equ', 'ite', 'mic', 'pro', 'sym', 'dep']


def evaluate_entities(samples, compare_type='entityMatch',
                      external_labels=None, keep_all_matches=False,
                      tolerance=2):
    """Evaluate entity-recognition predictions, accumulating TP/FP/FN per type.
    `compare_type` options:
      - 'indexMatch':       compare (start_idx, end_idx, type)
      - 'startIndexMatch':  compare (start_idx, entity, type)
      - 'endIndexMatch':    compare (end_idx, entity, type)
      - 'entityMatch':      compare only (entity, type)
    `external_labels`: optional ground-truth list (same length as samples).
        When provided, each element is a list of
        {'entity', 'start_idx', 'end_idx', 'type'} dicts and is used in
        preference to `sample['label']` (suited to BERT files' `entities`
        field); otherwise `sample['label']` is used (LLM files with an
        embedded mark-format label).
    `keep_all_matches`: when True, every occurrence of an LLM-predicted
        entity inside `original_text` is kept as a candidate position; when
        False (default), only the occurrence closest to the predicted
        position is kept, within `tolerance`.
    `tolerance`: position tolerance, default 2. Only effective when
        `keep_all_matches` is False.
    """
    stats = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in ENT_TYPES}
    stats['unknown'] = {'tp': 0, 'fp': 0, 'fn': 0}

    for idx, sample in enumerate(samples):
        ner_models = sample.get('ner_models', [])
        if ner_models:
            predict = ner_models[0].get('predict', '')
        else:
            predict = sample.get('predict', '')
            # Strip <think>/</think> wrappers emitted by reasoning models
            # such as DeepSeek / Qwen.
            if "</think>\n\n" in predict:
                predict = predict.split("</think>\n\n")[1]
            if "<think>\n" in predict:
                predict = predict.split("<think>\n")[1]
        label = sample.get('label', '')

        pred_entities = parse_marked_text_with_pos(predict)
        label_entities = parse_marked_text_with_pos(label)
        pred_entity_set = parse_marked_text_entity(predict)
        label_entity_set = parse_marked_text_entity(label)

        original_text = extract_original_text(sample)
        if original_text:
            pred_entities = _correct_to_original_text(
                pred_entities, original_text,
                tolerance=tolerance, keep_all_matches=keep_all_matches)
            if external_labels is None or external_labels[idx] is None:
                label_entities = _correct_to_original_text(
                    label_entities, original_text,
                    tolerance=tolerance, keep_all_matches=keep_all_matches)
            pred_entity_set = {k: v for k, v in pred_entity_set.items()
                               if v in original_text}
            label_entity_set = {k: v for k, v in label_entity_set.items()
                                if v in original_text}

        # Choose the source of ground truth
        use_external = (external_labels is not None
                        and external_labels[idx] is not None)
        if use_external:
            ext_ents = external_labels[idx]
            if compare_type == 'indexMatch':
                label_set = {(int(e['start_idx']), int(e['end_idx']), e['type'])
                             for e in ext_ents}
            elif compare_type == 'startIndexMatch':
                label_set = {(int(e['start_idx']), e['entity'], e['type'])
                             for e in ext_ents}
            elif compare_type == 'endIndexMatch':
                label_set = {(int(e['end_idx']), e['entity'], e['type'])
                             for e in ext_ents}
            else:
                label_set = {(e['entity'], e['type']) for e in ext_ents}
        else:
            if compare_type == 'indexMatch':
                label_set = {(l[0], l[1], l[3]) for l in label_entities}
            elif compare_type == 'startIndexMatch':
                label_set = {(l[0], l[2], l[3]) for l in label_entities}
            elif compare_type == 'endIndexMatch':
                label_set = {(l[1], l[2], l[3]) for l in label_entities}
            else:
                label_set = set(label_entity_set.keys())

        if compare_type == 'indexMatch':
            pred_set = {(p[0], p[1], p[3]) for p in pred_entities}
        elif compare_type == 'startIndexMatch':
            pred_set = {(p[0], p[2], p[3]) for p in pred_entities}
        elif compare_type == 'endIndexMatch':
            pred_set = {(p[1], p[2], p[3]) for p in pred_entities}
        else:
            pred_set = set(pred_entity_set.keys())

        for key in pred_set:
            etype = key[-1]
            if etype not in stats:
                continue
            if key in label_set:
                stats[etype]['tp'] += 1
            else:
                stats[etype]['fp'] += 1

        for key in label_set:
            etype = key[-1]
            if etype not in stats:
                continue
            if key not in pred_set:
                stats[etype]['fn'] += 1

    return stats


def calculate_metrics(stats):
    """Compute Precision/Recall/F1 per entity type from TP/FP/FN counts and
    return both macro and micro averages."""
    results = {}
    for ent_type in stats:
        tp = stats[ent_type]['tp']
        fp = stats[ent_type]['fp']
        fn = stats[ent_type]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0

        results[ent_type] = {
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'Support': tp + fn
        }

    total_tp = sum(v['tp'] for v in stats.values())
    total_fp = sum(v['fp'] for v in stats.values())
    total_fn = sum(v['fn'] for v in stats.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0

    results['MICRO_AVG'] = {
        'Precision': round(micro_precision, 4),
        'Recall': round(micro_recall, 4),
        'F1': round(micro_f1, 4),
        'Support': total_tp + total_fn
    }
    return results


def print_results(results):
    """Pretty-print a P/R/F1/Support table, per type plus macro/micro averages."""
    headers = ["Type", "Precision", "Recall", "F1", "Support"]
    print(f"{headers[0]:<10}{headers[1]:<12}{headers[2]:<12}"
          f"{headers[3]:<12}{headers[4]:<10}")

    total_Precision = 0
    total_Recall = 0
    total_F1 = 0
    total_Tab = 0

    for ent_type in sorted(results.keys()):
        if ent_type in ('MICRO_AVG', 'unknown'):
            continue
        m = results[ent_type]
        print(f"{ent_type:<10}{m['Precision']:<12.4f}{m['Recall']:<12.4f}"
              f"{m['F1']:<12.4f}{m['Support']:<10}")
        total_Precision += m['Precision']
        total_Recall += m['Recall']
        total_F1 += m['F1']
        total_Tab += 1

    if total_Tab > 0:
        print("\nMacro Average:")
        print(f"Precision: {total_Precision/total_Tab:.4f}")
        print(f"Recall:    {total_Recall/total_Tab:.4f}")
        print(f"F1:        {total_F1/total_Tab:.4f}")

    micro = results.get('MICRO_AVG', {})
    if micro:
        print("\nMicro Average:")
        print(f"Precision: {micro['Precision']:.4f}")
        print(f"Recall:    {micro['Recall']:.4f}")
        print(f"F1:        {micro['F1']:.4f}")
        print(f"Support:   {micro['Support']}")


# ====================================================================
# 5. Train/test split
# ====================================================================
def split_train_test(samples, test_size=2000, seed=42):
    """Randomly sample a test set with a fixed seed; the remainder is used
    as the training pool. Returns (train, test)."""
    n = len(samples)
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(np.arange(n), size=min(test_size, n), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = [i for i in range(n) if i not in test_idx_set]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in sorted(test_idx_set)]
    return train, test


# ====================================================================
# 6. Entry point
# ====================================================================
if __name__ == '__main__':
    eval_file = r'evl_f1\llm\glm4_ner_confidence_beams5.jsonl'
    samples = load_jsonl(eval_file)
    print(f"Loaded data: {len(samples)} records")

    compare_type = 'indexMatch'
    print(f"Match mode: {compare_type}")

    TRAIN_SAMPLE_SIZE = 4000
    TEST_SAMPLE_SIZE = 1000
    train, test = split_train_test(samples, test_size=TEST_SAMPLE_SIZE, seed=42)
    train = train[:TRAIN_SAMPLE_SIZE]
    print(f"Training pool: {len(train)} records, test set: {len(test)} records")

    stats = evaluate_entities(test, compare_type=compare_type)
    results = calculate_metrics(stats)
    print_results(results)
