from pathlib import Path
import sys
import grader.drgrpo_grader

def getPromptTemplate (prompt_filepath):
    with open(prompt_filepath, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template


if __name__ == '__main__':

    prompt_dir = "prompts/"
    prompt_filepath = prompt_dir + "r1_zero.prompt"

    prompt_template = getPromptTemplate (prompt_filepath)

    base_prompts = ["Compute 12 + 34*2 and explain.", "I have two apples today. I will eat one tomorrow. How many will I have the day after tomorrow?"]
    prompts = [prompt_template.format(question=ele) for ele in base_prompts]



    from vllm import LLM, SamplingParams

    model_id = "Qwen/Qwen2.5-Math-1.5B"  

    llm = LLM(
        model=model_id,
        #dtype="float16",   
        max_num_seqs=8
    )

    
    sampling_params = SamplingParams(max_tokens=16384, temperature=1.0, top_p = 1.0, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True #</answer> will be incl. in generation
    outputs = llm.generate(prompts, sampling_params)


    try :
        for o in outputs:
            print ("-----------"*3)
            print(o.outputs[0].text)
    finally:
        del llm
