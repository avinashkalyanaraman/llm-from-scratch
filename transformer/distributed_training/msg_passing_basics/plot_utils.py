import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def grouped_bar_plot(avg_timings):
    plt.close()
    
    grouped = defaultdict(dict)

    for (world_size, msg_size), timing in avg_timings.items():
        grouped[world_size][msg_size] = timing

    world_sizes = list(grouped.keys())
    msg_sizes = list(next(iter(grouped.values())).keys())

    x = np.arange(len(world_sizes))
    width = 0.8 / len(msg_sizes)

    plt.figure()

    for i, msg_size in enumerate(msg_sizes):
        values = [grouped[ws][msg_size] for ws in world_sizes]
        plt.bar(x + i * width, values, width, label=f"msg={msg_size}")

    plt.xticks(x + width * (len(msg_sizes) - 1) / 2, world_sizes)
    plt.xlabel("World Size")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()