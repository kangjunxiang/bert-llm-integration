"""
Split out the test set from the source data and save it to a new file.
"""
import json
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_FILE = os.path.join(BASE_DIR, 'data', 'CMeEE-V2', "llm",'CMeEE-V2_llm_mark_dev.json')
OUT_FILE = os.path.join(BASE_DIR, 'data', 'CMeEE-V2', "llm",'CMeEE-V2_llm_mark_test.json')

TRAIN_SAMPLE_SIZE = 4000  # Number of training (incl. validation) samples
TEST_SAMPLE_SIZE = 1000   # Number of test samples
SEED = 42


def split_train_test(samples, test_size=1000, seed=42):
    n = len(samples)
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(np.arange(n), size=min(test_size, n), replace=False)
    test_idx_set = set(test_idx.tolist())
    train_idx = [i for i in range(n) if i not in test_idx_set]
    train_llm = [samples[i] for i in train_idx]
    test_llm = [samples[i] for i in sorted(test_idx_set)]
    return train_llm, test_llm


def main():
    # Load JSON array (source file is standard JSON, not jsonl)
    with open(SRC_FILE, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'Loaded: {len(samples)} records')

    # Split
    train_llm, test_llm = split_train_test(
        samples, test_size=TEST_SAMPLE_SIZE, seed=SEED)
    # Cap training size (consistent with the original script)
    train_llm = train_llm[:TRAIN_SAMPLE_SIZE]
    test_samples = test_llm[:TEST_SAMPLE_SIZE]

    print(f'Training pool: {len(train_llm)} records')
    print(f'Test set: {len(test_samples)} records')

    # Save test set to a new file
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f'Test set saved to: {OUT_FILE}')


if __name__ == '__main__':
    main()
