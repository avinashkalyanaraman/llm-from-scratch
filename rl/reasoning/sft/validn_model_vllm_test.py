from vllm import SamplingParams
import utils, vllm_helper
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grader.drgrpo_grader


if __name__ == '__main__':

    SEED = 42

    data_file = 'data/sft.jsonl'
    torch.manual_seed (SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    #Model params!
    model_id = "Qwen/Qwen2.5-Math-1.5B"

    #Load the "base model" onto a separate GPU for validation.
    vllm_valdn_model = vllm_helper.init_vllm (model_id, device="cuda:0", seed=SEED)
    #Set sampling params for the validn model's generation!!
    sampling_params = SamplingParams(max_tokens=4096, temperature=1.0, top_p = 1.0, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True #</answer> will be incl. in generation


    #Read the validation dataset!
    #Read the dataset!
    dataset = utils.readJSONL (data_file)
    print (f"Total Dataset size = {len(dataset)}")

    #Split the data!
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)    
    #train_data = train_data[0:1000]
    #val_data = val_data[0:8]

    print (f"Train data len = {len(train_data)}")
    print (f"Val data len = {len(val_data)}")
    
    train_prompt_strs = [ele['prompt'] for ele in train_data]
    train_output_strs = [ele['response'] for ele in train_data]
    val_prompt_strs = [ele['prompt'] for ele in val_data]
    val_output_strs = [ele['response'] for ele in val_data]


    try :
        #2 Run generation on it with given validation prompts
        outputs = vllm_valdn_model.generate(val_prompt_strs, sampling_params)

        #3.Grade and see what is our validation result!
        results = utils.evaluate_model (grader.drgrpo_grader.r1_zero_reward_fn, outputs, val_output_strs)

        #4. Look at results to compute valdn_acc
        format_corrects_acc = len([ele for ele in results if ele[2]['format_reward'] > 0])*100./len(results)
        answer_corrects_acc = len([ele for ele in results if ele[2]['answer_reward'] > 0])*100./len(results)

        print(f"VALN :: The #format corrects = {format_corrects_acc:0.2f}, "
                f"#answer_corrects = {answer_corrects_acc:0.2f}")
    
    finally :
        del vllm_valdn_model