from pathlib import Path
import sys
import grader.drgrpo_grader
from datasets import load_dataset
import pickle

def getPromptTemplate (prompt_filepath):
    with open(prompt_filepath, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template

def rewritePrompt (sample, prompt_template):
    sample ['prompt'] = prompt_template.format(question=sample['problem'])
    return sample


def evaluate_vllm(llm, grader_fn, prompts, solutions, sampling_params):
    """
    Evaluate a language model on a list of prompts and compute evaluation metrics
    """
    outputs = llm.generate(prompts, sampling_params)
    results = []

    for o, solution, prompt in zip(outputs, solutions, prompts):
        generation = o.outputs[0].text
        result = grader_fn (generation, solution)
        results.append ( (prompt, solution, result) )

    return results

if __name__ == '__main__':

    prompt_dir = "prompts/"
    prompt_filepath = prompt_dir + "r1_zero.prompt"
    prompt_template = getPromptTemplate (prompt_filepath)

    #load the dataset
    dataset= load_dataset("qwedsacf/competition_math", "default")['train']
    #'train' is the hf-category being used!


    # Split 60% train, 40% test [7.5k , 5k split as desired!]
    split_dataset = dataset.train_test_split(test_size=0.4, seed=42)

    train_split = split_dataset["train"]
    test_split  = split_dataset["test"]


    #Set the prompt template in the desired format!
    train_split = train_split.map(rewritePrompt, fn_kwargs={"prompt_template": prompt_template}) #map : same function applied to all rows!
    test_split = test_split.map(rewritePrompt, fn_kwargs={"prompt_template": prompt_template})


    prompts = test_split ['prompt']
    solutions = test_split['solution']

    print (f"Total # of prompts = {len(prompts)}")
    print (f"{prompts[10]}")
    print ("---"*40)
    print (f"{solutions[10]}")


    from vllm import LLM, SamplingParams

    model_id = "Qwen/Qwen2.5-Math-1.5B"  

    llm = LLM(
        model=model_id,
        #dtype="float16",   
        max_num_seqs=128
    )

    
    sampling_params = SamplingParams(max_tokens=16384, temperature=1.0, top_p = 1.0, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True #</answer> will be incl. in generation
    
    try:
        outputs = evaluate_vllm (llm, grader.drgrpo_grader.r1_zero_reward_fn, prompts, solutions, sampling_params)

        with open("zero_shot_results.pkl", "wb") as f:
            pickle.dump(outputs, f)

    finally:
        del llm
