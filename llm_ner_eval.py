"""
LLM 命名实体识别 (NER) 评估脚本
======================================
针对 LLM 输出的 mark 格式标注 (etype : [entity]...) 评估 P/R/F1。
支持多种匹配模式: indexMatch / startIndexMatch / endIndexMatch / entityMatch。
可选启用位置容差 (tolerance) 与多位置候选 (keep_all_matches)。
"""

import json
import re
import ast
import numpy as np


# ====================================================================
# 1. 数据加载
# ====================================================================
def load_jsonl(filepath):
    """加载 jsonl 文件, 自动尝试 json / ast 解析, 跳过空行与解析失败行。"""
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
# 2. 原始文本抽取 (从 prompt 反推)
# ====================================================================
# 起始标记 (与本任务 prompt 模板的固定开头对齐)
_START_MARK = "明，与成人SARS相比，儿童[细胞下降]不明显，证明上述推测成立。\n"

# LLM chat 模板 end marker 候选, 按出现频率排序
_END_MARKS = [
    '<|eot_id|>',          # Llama2 / Qwen
    '<|end_of_text|>',     # Qwen
    '<|im_end|>',          # Qwen ChatML / GLM
    '<|end_turn|>',        # Gemma
    '<|assistant|>',       # GLM4
    '<｜Assistant｜>',      # DeepSeek (全角)
    '<|start_header_id|>', # Llama3
    '<|/assistant|>',      # Mistral
]


def get_promptPr(prompt, start_mark, end_mark):
    """从 prompt 中截取 start_mark 与 end_mark 之间的子串。"""
    if prompt is None:
        return None
    pattern = re.escape(start_mark) + '(.*?)' + re.escape(end_mark)
    match = re.search(pattern, prompt)
    return match.group(1) if match else None


def extract_original_text(sample):
    """从 sample 中提取原始待抽取文本。
    优先使用 text 属性, 否则从 prompt 中按多模板 end marker 抽取。
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
# 3. mark 格式标注解析
# ====================================================================
def _split_mark_line(line):
    """将 'etype : content' 拆成 (etype, content), 失败返回 None。"""
    line = line.strip()
    if not line or ' :' not in line:
        return None
    parts = line.split(' :', 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1]


def _entities_with_pos_in_content(content):
    """从单行 mark content 中抽取实体 (start, end, entity)。
    start/end 是行内偏移 (剔除 [] 后), entity 已 strip。
    兼容两种格式:
      A) 原句 + [entity] 标记 → 精确偏移
      B) 裸实体 (按 [、,;；] 切分) → 偏移在 content 内的实际位置
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
    """解析 mark 格式文本, 返回 [(start_idx, end_idx, entity, type, line_no), ...] 列表。
    start/end 是行内偏移 (剔除 [] 后); 后续需配合 _correct_to_original_text
    校正到 original_text 全文偏移。
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
    """解析 mark 格式文本, 返回 {(entity, type): entity} 字典。"""
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
    """将 mark 行内偏移校正到 original_text 全文中的真实位置。
    校正规则:
      1. 实体不在 original_text 中 → 丢弃 (LLM 幻觉, 不参与评估)
      2. 实体在 original_text 中, 位置已正确 → 保留
      3. 实体在 original_text 中, 位置不匹配:
         a. 仅 1 处匹配 → 直接使用该位置
         b. 多处匹配:
            - keep_all_matches=False: 启用位置容差 `tolerance`, 选容差内
              距离最近的位置; 若无, 退而选全部匹配中距离最近的位置
            - keep_all_matches=True: 全部采用, 把每一处匹配都作为候选位置加入
    """
    if not original_text:
        return entities
    corrected = []
    for start_idx, end_idx, entity, etype, line_no in entities:
        if not entity:
            continue
        # 1. 幻觉剔除
        if entity not in original_text:
            continue
        # 2. 位置已正确
        if 0 <= start_idx <= end_idx < len(original_text) \
                and original_text[start_idx:end_idx + 1] == entity:
            corrected.append((start_idx, end_idx, entity, etype, line_no))
            continue
        # 3. 搜索所有出现位置
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
# 4. 评估
# ====================================================================
ENT_TYPES = ['bod', 'dis', 'dru', 'equ', 'ite', 'mic', 'pro', 'sym', 'dep']


def evaluate_entities(samples, compare_type='entityMatch',
                      external_labels=None, keep_all_matches=False,
                      tolerance=2):
    """评估实体识别结果, 累计各类实体的 TP/FP/FN。
    compare_type 可选:
      - 'indexMatch':       比较 (start_idx, end_idx, type)
      - 'startIndexMatch':  比较 (start_idx, entity, type)
      - 'endIndexMatch':    比较 (end_idx, entity, type)
      - 'entityMatch':      只比较 (entity, type)
    external_labels: 可选外部 ground truth 列表 (与 samples 等长)。
                    若提供, 元素为 [{'entity', 'start_idx', 'end_idx', 'type'}, ...],
                    优先用此 ground truth (适合 BERT 文件 entities 字段);
                    否则用 sample['label'] 字段 (LLM 文件内置 label, mark 格式)。
    keep_all_matches: True 时 LLM 预测实体在 original_text 中出现多次会被全部
                    保留为候选位置; False 时按 tolerance 选最近 1 处 (默认)。
    tolerance: 位置容差, 默认 2。仅在 keep_all_matches=False 时生效。
    """
    stats = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in ENT_TYPES}
    stats['unknown'] = {'tp': 0, 'fp': 0, 'fn': 0}

    for idx, sample in enumerate(samples):
        ner_models = sample.get('ner_models', [])
        if ner_models:
            predict = ner_models[0].get('predict', '')
        else:
            predict = sample.get('predict', '')
            # 去除 DeepSeek/Qwen 等推理模型 <think>/</think> 包裹
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

        # 决定 ground truth 来源
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
    """根据 TP/FP/FN 计算各类实体的 Precision/Recall/F1, 并给出宏/微平均。"""
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
    """格式化输出 P/R/F1/Support 表格 (按类别 + 宏/微平均)。"""
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
# 5. 训练/测试切分
# ====================================================================
def split_train_test(samples, test_size=2000, seed=42):
    """按固定种子随机切分测试集, 剩余作为训练集。返回 (train, test)。"""
    n = len(samples)
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(np.arange(n), size=min(test_size, n), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = [i for i in range(n) if i not in test_idx_set]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in sorted(test_idx_set)]
    return train, test


# ====================================================================
# 6. 入口
# ====================================================================
if __name__ == '__main__':
    eval_file = r'evl_f1\llm\glm4_ner_confidence_beams5.jsonl'
    samples = load_jsonl(eval_file)
    print(f"加载数据: {len(samples)} 条")

    compare_type = 'indexMatch'
    print(f"匹配模式: {compare_type}")

    TRAIN_SAMPLE_SIZE = 4000
    TEST_SAMPLE_SIZE = 1000
    train, test = split_train_test(samples, test_size=TEST_SAMPLE_SIZE, seed=42)
    train = train[:TRAIN_SAMPLE_SIZE]
    print(f"训练池: {len(train)} 条, 测试集: {len(test)} 条")

    stats = evaluate_entities(test, compare_type=compare_type)
    results = calculate_metrics(stats)
    print_results(results)
