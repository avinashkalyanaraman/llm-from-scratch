import wandb
import os


def get_run(run_id_file = "wandb_run_id.txt"):
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        return run_id
    else:
        return None
    

def set_run(run_id_file, run_id):
    # If this is a new run, save its ID for next time
    with open(run_id_file, "w") as f:
        f.write(run_id)
