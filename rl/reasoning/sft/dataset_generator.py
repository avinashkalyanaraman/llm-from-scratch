#!/usr/bin/env python3
import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------- CONFIG ----------------
#DATA_FILE = "hf://datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_sample_1k.jsonl"
# for real runs, switch to:
DATA_FILE = "hf://datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M.jsonl.zst"

MATH_SOURCES = {
    "openR1Math_default",
    "openR1Math_extended",
    "NuminaMath_1.5",
    "MetaMathQA",
}

MAX_THINK_TOKENS = 1024    # cap reasoning length
N_EXAMPLES_TO_SHOW = 10000   # how many to print

# MCQ detection patterns
MCQ_PATTERNS = [
    r"^[A-D]\.",        # "A." or "B."
    r"^\([A-D]\)",      # "(A)" or "(B)"
    r"which of the following",
    r"choose the correct",
    r"select the correct",
    r"the correct statement",
    r"options?:"
]

# re.M = multiline, re.I = case-insensitive
MCQ_RE = re.compile("|".join(MCQ_PATTERNS), flags=re.M | re.I)



def getPromptTemplate (prompt_filepath):
    with open(prompt_filepath, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template

#Rewrite prompt in r1_zero format!
def rewritePrompt(user_text, prompt_template):
    return prompt_template.format(question=user_text)


def is_mcq_prompt(text: str) -> bool:
    """Return True if prompt looks like multiple-choice."""
    return bool(MCQ_RE.search(text))


def extract_think_and_answer(assistant_text: str):
    """Return (think, answer) or (None, None) if not found."""
    m = re.search(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", assistant_text, flags=re.S)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def getExample (user_text, think_text, answer_text):
    think_text = think_text + "</think>"
    answer_text = "<answer>" + answer_text + "</answer>"
    response = think_text + answer_text
    output = {"prompt" : user_text, "response" : response}
    return output

def writeOutput (output_filepath, examples):
    """Write a list of dicts (with 'prompt' and 'response') to a JSONL file."""
    with open(output_filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(examples)} examples to {output_filepath}")


def main():
    # tokenizer only used to count tokens in <think>
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    #Get the user-prompt template!
    prompt_dir = "../prompts/"
    prompt_filepath = prompt_dir + "r1_zero.prompt"
    prompt_template = getPromptTemplate (prompt_filepath)

    #Output file to write to!
    output_dir = 'data/'
    output_filepath = 'sft.jsonl'

    isPrint = False

    # load as PLAIN JSON to avoid Arrow schema errors
    ds = load_dataset(
        "json",
        data_files=DATA_FILE,
        split="train",
        streaming=True,
    )

    shown = 0
    examples = []
    for ex in ds:
        # basic structure from the AM dataset
        msgs = ex.get("messages")
        if not msgs or len(msgs) < 2:
            continue

        user_msg = msgs[0]
        asst_msg = msgs[1]

        user_text = user_msg.get("content", "")
        asst_text = asst_msg.get("content", "")

        # 1) keep only math-ish sources
        user_info = user_msg.get("info") or {}
        src = user_info.get("source", "")
        if src not in MATH_SOURCES:
            continue

        # 2) skip MCQs
        if is_mcq_prompt(user_text):
            continue

        # 3) extract think/answer
        think, answer = extract_think_and_answer(asst_text)
        if think is None:
            continue

        # 4) length filter on <think>
        think_ids = tokenizer(think, add_special_tokens=False).input_ids
        if len(think_ids) > MAX_THINK_TOKENS:
            # you can either skip OR keep the tail; here we skip to keep it simple
            # to keep the tail, uncomment below:
            # tail_ids = think_ids[-MAX_THINK_TOKENS:]
            # think = tokenizer.decode(tail_ids, skip_special_tokens=True)
            continue

        #Rewrite user_text in the desired r1_zero prompt
        user_text = rewritePrompt (user_text.strip(), prompt_template)


        # ----- passed all filters: print -----
        shown += 1
        # compact whitespace in reasoning for display
        clean_think = " ".join(think.split())
        
        if isPrint:
            print(f"\n===== Example {shown} =====")
            print("Prompt:")
            print(user_text.strip())
            print("\nReasoning:")
            print(clean_think)
            print("\nFinal Answer:")
            print(answer.strip())


        example = getExample (user_text.strip(), clean_think, answer.strip())
        examples.append (example)

        if shown % 25 == 0:
            print (f"# of examples handled = {shown}")

        if shown >= N_EXAMPLES_TO_SHOW:
            break

    if shown == 0:
        print("No matching math examples found with current filters. Try:")
        print("- lowering MAX_THINK_TOKENS")
        print("- removing MCQ filter")
        print("- switching to the bigger file: am_0.9M.jsonl.zst")
    
    writeOutput (output_filepath, examples)



if __name__ == "__main__":
    main()
