"""
拆分出测试集数据并保存到新文件。
"""
import json
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_FILE = os.path.join(BASE_DIR, 'data', 'CMeEE-V2', "llm",'CMeEE-V2_llm_mark_dev.json')
OUT_FILE = os.path.join(BASE_DIR, 'data', 'CMeEE-V2', "llm",'CMeEE-V2_llm_mark_test.json')

TRAIN_SAMPLE_SIZE = 4000  # 训练 (含验证) 样本数 
TEST_SAMPLE_SIZE = 1000   # 测试样本数 
SEED = 42


def split_train_test(samples, test_size=1000, seed=42):
    """与 llm_ner_eval_Temp_mark.py 中的实现完全一致。"""
    n = len(samples)
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(np.arange(n), size=min(test_size, n), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = [i for i in range(n) if i not in test_idx_set]
    train_llm = [samples[i] for i in train_idx]
    test_llm = [samples[i] for i in sorted(test_idx_set)]
    return train_llm, test_llm


def main():
    # 加载 JSON 数组 (源文件是标准 JSON, 不是 jsonl)
    with open(SRC_FILE, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'加载完成: {len(samples)} 条')

    # 划分
    train_llm, test_llm = split_train_test(
        samples, test_size=TEST_SAMPLE_SIZE, seed=SEED)
    # 控制训练样本数 (与原脚本一致)
    train_llm = train_llm[:TRAIN_SAMPLE_SIZE]
    test_samples = test_llm[:TEST_SAMPLE_SIZE]

    print(f'训练池: {len(train_llm)} 条')
    print(f'测试集: {len(test_samples)} 条')

    # 保存测试集到新文件
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f'测试集已保存到: {OUT_FILE}')


if __name__ == '__main__':
    main()
