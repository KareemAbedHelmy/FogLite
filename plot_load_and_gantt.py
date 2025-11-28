import os
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

from config.nodes_config import NODES_CONFIG
from env_fog import FogEnv
from env_wrapper import FogGymWrapper
from baselines import round_robin_policy, least_loaded_policy


def random_policy(step_idx: int, num_nodes: int, state: np.ndarray) -> int:
    return np.random.randint(0, num_nodes)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_fog_env() -> FogEnv:
    return FogEnv(
        nodes_config=NODES_CONFIG,
        episode_length=500,
        alpha=0.4,
        beta=0.6,
        lambda_deadline=4.0,
        lambda_overload=0.5,
        u_max=0.9,
        e_ref=20.0,
        l_ref=2.0,
        task_length_range=(50.0, 250.0),
        deadline_slack_range=(1.0, 3.0),
        interarrival_range=(0.0, 0.05),
        use_task_dependence=True,
        max_stages_per_job=3,
        handoff_latency=0.2,
        seed=123,
    )


def run_heuristic_with_log(
    policy_name: str,
    policy_fn,
    episodes: int = 1,
) -> Dict:
    fog_env = build_fog_env()
    node_task_counts = Counter()
    last_timeline = []

    for ep in range(episodes):
        state = fog_env.reset()
        done = False
        step_idx = 0

        while not done:
            action = policy_fn(step_idx, fog_env.num_nodes, state)
            next_state, reward, done, info = fog_env.step(action)
            state = next_state if next_state is not None else state
            step_idx += 1

        for entry in fog_env.timeline_log:
            node_task_counts[entry["node_id"]] += 1

        last_timeline = list(fog_env.timeline_log)

    return {"counts": node_task_counts, "timeline": last_timeline, "env": fog_env}


def run_dqn_with_log(
    model_path: str,
    episodes: int = 1,
) -> Dict:
    fog_env = build_fog_env()
    wrapped_env = FogGymWrapper(fog_env)
    model = DQN.load(model_path, env=wrapped_env)

    node_task_counts = Counter()
    last_timeline = []

    for ep in range(episodes):
        obs, _ = wrapped_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, info = wrapped_env.step(action)
            obs = next_obs

        for entry in fog_env.timeline_log:
            node_task_counts[entry["node_id"]] += 1

        last_timeline = list(fog_env.timeline_log)

    return {"counts": node_task_counts, "timeline": last_timeline, "env": fog_env}


def plot_load_distribution(
    load_counts_by_policy: Dict[str, Counter],
    node_names: Dict[int, str],
):
    import numpy as np

    policies = list(load_counts_by_policy.keys())
    node_ids = sorted(node_names.keys())

    data = []
    for pname in policies:
        counts = load_counts_by_policy[pname]
        data.append([counts.get(nid, 0) for nid in node_ids])

    x = np.arange(len(node_ids))
    width = 0.18

    plt.figure()
    for i, pname in enumerate(policies):
        plt.bar(x + i * width, data[i], width=width, label=pname)

    plt.xticks(x + width * (len(policies) - 1) / 2, [node_names[n] for n in node_ids], rotation=20)
    plt.ylabel("Number of tasks")
    plt.title("Load distribution across nodes")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "load_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved load distribution plot to {out_path}")


def plot_gantt_for_job(timeline: List[Dict], job_id, title_suffix: str):
    job_entries = [e for e in timeline if e["job_id"] == job_id]
    if not job_entries:
        print(f"No entries for job_id={job_id}")
        return

    job_entries.sort(key=lambda e: e["stage_idx"])

    fig, ax = plt.subplots(figsize=(8, 3))

    yticks = []
    ylabels = []

    for idx, entry in enumerate(job_entries):
        start = entry["start_time"]
        finish = entry["finish_time"]
        duration = finish - start
        node_name = entry["node_name"]

        y = idx
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8), label=node_name)

        yticks.append(y)
        ylabels.append(f"Stage {entry['stage_idx']} ({node_name})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Workflow timeline for job {job_id} ({title_suffix})")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"gantt_job_{job_id}_{title_suffix}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Gantt chart to {out_path}")


def main():
    policies = {
        "RoundRobin": round_robin_policy,
        "LeastLoaded": least_loaded_policy,
        "Random": random_policy,
    }

    load_counts_by_policy = {}
    timelines_by_policy = {}
    envs_by_policy = {}

    for pname, pfn in policies.items():
        print(f"Evaluating heuristic policy: {pname}")
        result = run_heuristic_with_log(pname, pfn, episodes=1)
        load_counts_by_policy[pname] = result["counts"]
        timelines_by_policy[pname] = result["timeline"]
        envs_by_policy[pname] = result["env"]

    print("Evaluating DQN policy")
    dqn_result = run_dqn_with_log(
        model_path=os.path.join(BASE_DIR, "dqn_fog_scheduler.zip"),
        episodes=1,
    )
    load_counts_by_policy["DQN"] = dqn_result["counts"]
    timelines_by_policy["DQN"] = dqn_result["timeline"]
    envs_by_policy["DQN"] = dqn_result["env"]

    any_env = next(iter(envs_by_policy.values()))
    node_names = {
        node.node_id: getattr(node, "name", f"Node{node.node_id}")
        for node in any_env.nodes
    }

    plot_load_distribution(load_counts_by_policy, node_names)

    dqn_timeline = timelines_by_policy["DQN"]
    if dqn_timeline:
        first_job_id = dqn_timeline[0]["job_id"]
        plot_gantt_for_job(dqn_timeline, job_id=first_job_id, title_suffix="DQN")
    else:
        print("No tasks in DQN timeline; Gantt chart not generated.")


if __name__ == "__main__":
    main()
