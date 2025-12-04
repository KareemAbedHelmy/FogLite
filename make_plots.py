import os
import matplotlib.pyplot as plt

# Create plots directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

algos = ["Round Robin", "Least Loaded", "Random", "DQN"]

# === CORRECTED RESULTS FROM eval_policies.py ===
# Per-task metrics
energy_task  = [17.5749, 13.3485, 16.2220, 11.1944]   # Avg energy per task (J)
latency_task = [1.3245, 1.5163, 1.3297, 1.0432]       # Avg latency per task (s)
miss_task    = [12.34, 16.58, 12.37, 4.74]            # Deadline miss rate per task (%)

# Per-job metrics
energy_job   = [35.1005, 26.6781, 32.4675, 22.2556]   # Avg energy per job (J)
latency_job  = [1.3678, 1.7579, 1.3913, 1.1230]       # Avg latency per job (s)
miss_job     = [11.94, 20.19, 12.19, 4.52]            # Job deadline miss rate (%)


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