import wandb

def logToWANDB (run, key, val, x_val, IS_WANDB):
    if IS_WANDB:        
        run.log ({key : val}, step = x_val)


def logDictToWANDB (run, log_dict, x_val, IS_WANDB):
    if IS_WANDB:        
        run.log (log_dict, step = x_val)

def logToWANDBWithStepKey(run, key, val, step_key, step_val, global_step_val, IS_WANDB ):
    if IS_WANDB:   
        run.log({step_key: step_val, key: val}, step = global_step_val)
