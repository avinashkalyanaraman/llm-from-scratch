#Starter code from cs336

from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
import torch
from unittest.mock import patch


#This places a base-model on the GPU.
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    13
    """

    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch( 
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        )


#This replaces the already placed "base" mode with a given set of weights! 
#Since we use torch.compile (model) the below breaks. Hence use "load_policy_into_vllm_instance"
def load_policy_into_vllm_instance_orig(policy, llm):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """

    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def load_policy_into_vllm_instance(policy, llm):
    # HF model weights
    policy_sd = policy.state_dict()

    # vLLM underlying model
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # names vLLM actually has
    vllm_names = set(n for n, _ in vllm_model.named_parameters())
    vllm_names.update(n for n, _ in vllm_model.named_buffers())

    filtered = []
    dropped = []
    for name, tensor in policy_sd.items():
        if name in vllm_names:
            filtered.append((name, tensor))
        else:
            dropped.append(name)

    # optional: make sure we only dropped bookkeeping keys
    unexpected = [k for k in dropped if not k.startswith("_orig_mod")]
    if unexpected:
        # this means architectures really differ
        raise ValueError(f"dropping unexpected keys when loading into vLLM: {unexpected}")

    vllm_model.load_weights(filtered)


def load_policy_into_vllm_instance2(policy, llm):
    # 1) unwrap if torch.compile() was applied
    base_policy = getattr(policy, "_orig_mod", policy)

    # 2) (optional but helpful) move to CPU to avoid extra VRAM spikes
    #    and ensure tensors are CPU when handing them to vLLM.
    #base_policy = base_policy.to("cpu")

    # 3) grab state dict
    sd = base_policy.state_dict()  # keys now match vLLM (no "_orig_mod." prefix)

    # 4) get vLLM model
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # 5) load (vLLM expects an iterable of (name, tensor))
    #vllm_model.load_weights((k, v.cpu()) for k, v in sd.items())
    vllm_model.load_weights(sd.items())


    #Sanity check if copy was okay!
    vllm_names = {n for n, _ in vllm_model.named_parameters()} | {n for n, _ in vllm_model.named_buffers()}
    missing = [k for k in sd.keys() if k not in vllm_names]
    assert not missing, f"Missing in vLLM: {missing[:10]}"

