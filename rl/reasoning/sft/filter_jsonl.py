'''
This code takes as input 
    (i)jsonl file
    (ii)pickle file that has a list of line numbers to keep

And,
    stores only those lines to a new jsonl file
'''

import pickle
import json

if __name__ == '__main__':

    input_file = 'data/sft.jsonl'
    pickle_file = 'data/correct_answers_indices.pkl'
    output_file = 'data/sft_filtered.jsonl'

    # 1. Read the pickle file (list of line numbers to keep)
    with open(pickle_file, "rb") as f:
        keep_indices = set(pickle.load(f))   # convert to set for faster lookup

    # 2. Read jsonl and filter
    filtered = []
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in keep_indices:
                obj = json.loads(line)
                filtered.append(obj)

    # 3. Write filtered lines to new jsonl
    with open(output_file, "w", encoding="utf-8") as f:
        for obj in filtered:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(filtered)} lines to {output_file}")