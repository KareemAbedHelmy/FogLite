import os
import matplotlib.pyplot as plt

# Create plots directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

algos = ["Round Robin", "Least Loaded", "Random", "DQN"]

# === FINAL RESULTS FROM eval_policies.py ===
# Per-task metrics
energy_task  = [18.5392, 18.7305, 18.7178, 19.9546]   # Avg energy per task (J)
latency_task = [5.5166, 1.3520, 7.4598, 1.4769]       # Avg latency per task (s)
miss_task    = [44.11, 11.92, 58.83, 12.71]           # Deadline miss rate per task (%)

# Per-job metrics
energy_job   = [37.0764, 37.7211, 37.2734, 40.6445]   # Avg energy per job (J)
latency_job  = [7.4748, 1.3491, 10.7793, 1.5079]      # Avg latency per job (s)
miss_job     = [51.30, 10.16, 72.11, 10.97]           # Job deadline miss rate (%)


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
