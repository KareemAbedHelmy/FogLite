import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
@dataclass
class Task:
    """
    Represents one incoming task.
    length_mi: task length in million instructions
    deadline: absolute time by which it should finish
    arrival_time: when it arrived in the system
    """
    length_mi: float
    deadline: float
    arrival_time: float
@dataclass
class Node:
    """
    Represents one fog or cloud node.
    """
    node_id: int
    name: str
    is_cloud: bool
    mips: float           # million instructions per second
    cores: int
    power_idle: float     # Watts
    power_max: float      # Watts at 100% utilisation
    base_latency: float   # network latency (seconds) from IoT to this node
    # dynamic fields
    running_intervals: List[Tuple[float, float]] = field(default_factory=list)  # (task, completion_time)
    queue: List[Task] = field(default_factory=list)
    busy_until: float = 0.0  # time when node becomes free
class FogEnv:
    """
    Custom fog scheduling environment.
    - At each step, a new task arrives (Task object).
    - Agent chooses a node index (0..num_nodes-1).
    - Env simulates assignment, computes latency & energy.
    - Returns next_state, reward, done, info.
    """
    def __init__(
        self,
        nodes_config: List[Dict[str, Any]],
        episode_length: int = 500,
        alpha: float = 0.5,
        beta: float = 0.5,
        lambda_deadline: float = 2.0,
        lambda_overload: float = 0.5,
        u_max: float = 0.9,
        e_ref: float = 6.0,
        l_ref: float = 3.0,
        task_length_range: Tuple[float, float] = (50.0, 250.0),  # MI
        deadline_slack_range: Tuple[float, float] = (1.0, 3.0),  # seconds
        interarrival_range: Tuple[float, float] = (0.0, 0.05),  # seconds
        seed: Optional[int] = None,
    ):
        # reward weights
        self.alpha = alpha
        self.beta = beta
        self.lambda_deadline = lambda_deadline
        self.lambda_overload = lambda_overload
        self.interarrival_range = interarrival_range
        self.u_max = u_max
        # normalisation constants
        self.E_ref = e_ref
        self.L_ref = l_ref
        # episode settings
        self.episode_length = episode_length
        self.task_length_range = task_length_range
        self.deadline_slack_range = deadline_slack_range
        # RNG
        self._rng = np.random.default_rng(seed)
        # Build nodes
        self.nodes: List[Node] = []
        for i, cfg in enumerate(nodes_config):
            self.nodes.append(
                Node(
                    node_id=i,
                    name=cfg.get("name", f"Node{i}"),
                    is_cloud=cfg.get("is_cloud", False),
                    mips=float(cfg["mips"]),
                    cores=int(cfg["cores"]),
                    power_idle=float(cfg["power_idle"]),
                    power_max=float(cfg["power_max"]),
                    base_latency=float(cfg["base_latency"]),
                )
            )
        self.num_nodes = len(self.nodes)
        # Simulation state
        self.current_time: float = 0.0
        self.steps_done: int = 0
        self.current_task: Optional[Task] = None
        
    def reset(self) -> np.ndarray:
        """
        Reset environment for a new episode.
        Returns initial state.
        """
        self.current_time = 0.0
        self.steps_done = 0
        for node in self.nodes:
            node.queue.clear()
            node.running_intervals.clear()
            node.busy_until = 0.0

        self.current_task = self._generate_task()
        return self._build_state()

    def step(self, action: int):
        """
        Perform one scheduling decision.
        Args:
            action: integer node index (0 <= action < num_nodes)
        Returns:
            next_state (np.ndarray or None if done),
            reward (float),
            done (bool),
            info (dict with metrics)
        """
        if not (0 <= action < self.num_nodes):
            raise ValueError(f"Invalid action {action}, num_nodes={self.num_nodes}")

        if self.current_task is None:
            raise RuntimeError("No current task. Did you call reset()?")

        chosen_node = self.nodes[action]
        task = self.current_task
        # simulate assignment
        E_i, L_i, U_i, deadline_miss, overload = self._simulate_task_on_node(task, chosen_node)
        # reward
        reward = self._compute_reward(
            E_i=E_i,
            L_i=L_i,
            D_i=task.deadline,
            U_i=U_i,
            deadline_miss=deadline_miss,
            overload=overload,
            arrival=task.arrival_time,
        )
        # simple model: next task arrives immediately after this decision
        self.steps_done += 1
        done = self.steps_done >= self.episode_length

        if not done:
            # Advance time by random inter-arrival before the next task arrives
            delta_t = float(self._rng.uniform(*self.interarrival_range))
            self.current_time = max(self.current_time, task.arrival_time) + delta_t
            # and then generate next task at new current_time
            self.current_task = self._generate_task()
            next_state = self._build_state()
        else:
            self.current_task = None
            next_state = None

        info = {
            "E_i": E_i,
            "L_i": L_i,
            "U_i": U_i,
            "deadline_miss": deadline_miss,
            "overload": overload,
            "chosen_node": chosen_node.name,
        }

        return next_state, reward, done, info
    
    def _generate_task(self) -> Task:
        """
        Generate a new task with random length and deadline slack.
        """
        length = float(self._rng.uniform(*self.task_length_range))
        slack = float(self._rng.uniform(*self.deadline_slack_range))

        arrival = self.current_time  # one task per step at current time
        deadline = arrival + slack

        return Task(
            length_mi=length,
            deadline=deadline,
            arrival_time=arrival,
        )
    def _simulate_task_on_node(
    self,
    task: Task,
    node: Node,
) -> Tuple[float, float, float, bool, bool]:
        """
        Compute latency, energy and utilisation impact of assigning `task` to `node`,
        using continuous-time energy integration.
        - We track all (start_time, end_time) intervals of tasks on this node.
        - For energy, we integrate power over time only over intervals where THIS task
        is active, with power determined by total concurrent tasks.
        """
        # Task cannot start before it arrives or before node is free
        start_time = max(task.arrival_time, node.busy_until)
        # Service time (seconds) = length (MI) / speed (MI/s)
        service_time = task.length_mi / node.mips
        end_time = start_time + service_time
        # Record this task's interval on the node
        node.running_intervals.append((start_time, end_time))
        # Update busy_until to reflect that the node is busy until at least end_time
        node.busy_until = max(node.busy_until, end_time)
        # keep queue list for stats (not needed for energy directly)
        node.queue.append(task)
        net_delay = node.base_latency
        # End-to-end latency from arrival, including network delay
        L_i = (end_time - task.arrival_time) + net_delay
        # Collect all time boundaries from running intervals
        boundaries = []
        for s, e in node.running_intervals:
            boundaries.append(s)
            boundaries.append(e)
        boundaries = sorted(set(boundaries))
        E_i = 0.0
        max_util_during_task = 0.0
        overload = False
        # this task's active interval
        t_task_start = start_time
        t_task_end = end_time

        for i in range(len(boundaries) - 1):
            t_start = boundaries[i]
            t_end = boundaries[i + 1]
            duration = t_end - t_start
            if duration <= 0:
                continue
            # Checks if the sub-interval overlaps with this tasks lifetime
            if t_end <= t_task_start or t_start >= t_task_end:
                # No overlap with our task -> this energy is for other tasks, ignore
                continue
            # Overlap portion with this task (in case the segment extends beyond)
            seg_start = max(t_start, t_task_start)
            seg_end = min(t_end, t_task_end)
            seg_duration = seg_end - seg_start
            if seg_duration <= 0:
                continue
            # Count how many tasks are active on the node during [t_start, t_end)
            active = 0
            for s, e in node.running_intervals:
                if not (e <= t_start or s >= t_end):  # intervals overlap
                    active += 1
                    
            # Utilisation based on number of active tasks vs cores
            U = min(1.0, active / max(1, node.cores))
            # Linear power model
            P = node.power_idle + (node.power_max - node.power_idle) * U
            # Energy contribution for THIS task in this overlapping subinterval
            E_i += P * seg_duration
            # Track max utilisation while this task is active
            if U > max_util_during_task:
                max_util_during_task = U
            # Overload if utilisation exceeds threshold U_max while this task is active
            if U > self.u_max:
                overload = True
        # Check for deadline miss
        deadline_slack = task.deadline - task.arrival_time
        deadline_miss = L_i > deadline_slack
        # Use max utilisation during task as U_i for reward
        U_i = max_util_during_task
        return E_i, L_i, U_i, deadline_miss, overload

    def _compute_reward(
        self,
        E_i: float,
        L_i: float,
        D_i: float,
        U_i: float,
        deadline_miss: bool,
        overload: bool,
        arrival: float,
    ) -> float:
        """
        Apply the RL reward formula:
        r_i = - (alpha * E_hat + beta * L_hat)
              - lambda_deadline * I[deadline_miss]
              - lambda_overload * I[overload]
        """
        # normalise
        hat_E = E_i / self.E_ref if self.E_ref > 0 else E_i
        hat_L = L_i / self.L_ref if self.L_ref > 0 else L_i
        cost = self.alpha * hat_E + self.beta * hat_L
        penalty = 0.0
        if deadline_miss:
            penalty += self.lambda_deadline
        if overload:
            penalty += self.lambda_overload
        reward = -(cost + penalty)
        return reward

    def _build_state(self) -> np.ndarray:
        """
        Build state vector:
        [ node1_feature, node2_feature, ..., nodeN_feature,
          task_length_norm, task_slack_norm ]
        For now, each node feature is (busy_until - current_time) normalised.
        """
        node_feats = []
        # normalisation factor for how far in the future busy_until can go
        max_future = max(self.deadline_slack_range[1], 1.0)
        for node in self.nodes:
            avail_in = max(0.0, node.busy_until - self.current_time)
            node_feats.append(avail_in / max_future)

        t = self.current_task
        slack = max(0.0, t.deadline - self.current_time)
        length_norm = t.length_mi / max(self.task_length_range)
        slack_norm = slack / max(self.deadline_slack_range)
        state = np.array(node_feats + [length_norm, slack_norm], dtype=np.float32)
        return state