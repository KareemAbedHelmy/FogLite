import os
import matplotlib.pyplot as plt

# Create plots directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

algos = ["Round Robin", "Least Loaded", "Random", "DQN"]
energy_task = [29.6535, 19.2552, 29.3158, 18.3908]
latency_task = [10.3900, 2.4376, 10.0219, 3.8739]
miss_task = [75.30, 60.04, 74.70, 49.08]

energy_job = [73.8239, 48.1835, 73.2101, 44.9371]
latency_job = [16.4544, 2.4789, 15.5755, 4.1890]
miss_job = [92.52, 60.43, 91.75, 53.81]

def save_bar(y, title, ylabel, filename):
    plt.figure()
    plt.bar(algos, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = os.path.join(plots_dir, filename)
    plt.savefig(path)
    plt.close()
    return path

paths = {}

paths['energy_per_task'] = save_bar(
    energy_task,
    "Average Energy per Task",
    "Energy (J)",
    "energy_per_task.png"
)

paths['latency_per_task'] = save_bar(
    latency_task,
    "Average Latency per Task",
    "Latency (s)",
    "latency_per_task.png"
)

paths['deadline_miss_task'] = save_bar(
    miss_task,
    "Task Deadline Miss Rate",
    "Miss Rate (%)",
    "deadline_miss_rate_task.png"
)

paths['energy_per_job'] = save_bar(
    energy_job,
    "Average Energy per Job",
    "Energy (J)",
    "energy_per_job.png"
)

paths['latency_per_job'] = save_bar(
    latency_job,
    "Average Latency per Job",
    "Latency (s)",
    "latency_per_job.png"
)

paths['deadline_miss_job'] = save_bar(
    miss_job,
    "Job Deadline Miss Rate",
    "Miss Rate (%)",
    "deadline_miss_rate_job.png"
)

# Scatter plots for energy vs latency
def save_scatter(latency, energy, title, xlab, ylab, filename):
    plt.figure()
    for x, y, label in zip(latency, energy, algos):
        plt.scatter(x, y)
        plt.text(x, y, label, fontsize=8, ha='right', va='bottom')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    path = os.path.join(plots_dir, filename)
    plt.savefig(path)
    plt.close()
    return path

paths['scatter_task'] = save_scatter(
    latency_task,
    energy_task,
    "Energy vs Latency per Task",
    "Latency (s)",
    "Energy (J)",
    "energy_vs_latency_task.png"
)

paths['scatter_job'] = save_scatter(
    latency_job,
    energy_job,
    "Energy vs Latency per Job",
    "Latency (s)",
    "Energy (J)",
    "energy_vs_latency_job.png"
)

paths
