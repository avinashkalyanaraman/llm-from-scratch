from vllm import SamplingParams
import utils, vllm_helper
from sklearn.model_selection import train_test_split
import torch
import pickle

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grader.drgrpo_grader


if __name__ == '__main__':

    IS_FILTERING = False


    SEED = 42

    data_file = 'data/sft_train.jsonl'
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


    prompt_strs = [ele['prompt'] for ele in dataset]
    output_strs = [ele['response'] for ele in dataset]
    

    try :
        #2 Run generation on it with given validation prompts
        outputs = vllm_valdn_model.generate(prompt_strs, sampling_params)

        #3.Grade and see what is our validation result!
        results = utils.evaluate_model (grader.drgrpo_grader.r1_zero_reward_fn, outputs, output_strs)

        #4. Look at results to compute valdn_acc
        format_corrects_acc = len([ele for ele in results if ele[2]['format_reward'] > 0])*100./len(results)
        answer_corrects_acc = len([ele for ele in results if ele[2]['answer_reward'] > 0])*100./len(results)

        print(f"VALN :: The #format corrects = {format_corrects_acc:0.2f}, "
                f"#answer_corrects = {answer_corrects_acc:0.2f}")
        
        #This is for filtering the right indices to sft on only the correct answers!
        correct_answers_indices = [ii for ii, ele in enumerate(results) if ele[2]['answer_reward'] > 0]

        if IS_FILTERING:
            with open("data/correct_answers_indices.pkl", "wb") as f:
                pickle.dump(correct_answers_indices, f)
    
    finally :
        del vllm_valdn_model