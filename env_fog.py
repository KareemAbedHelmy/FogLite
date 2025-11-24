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
    job_id: identifier of the job this task belongs to
    stage_idx: index of the stage in the job
    num_stages: total number of stages in the job
    """
    length_mi: float
    deadline: float
    arrival_time: float
    job_id: int = 0
    stage_idx: int = 0
    num_stages: int = 1
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
        use_task_dependence: bool = False,
        max_stages_per_job: int = 3,
        handoff_latency: float = 0.2, # seconds penalty for handing off between nodes in a job
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
        #task dependency controls
        self.use_task_dependence = use_task_dependence
        self.max_stages_per_job = max_stages_per_job
        self.next_job_id = 0 # should auto increment for each new job
        self.deadline_ref = deadline_slack_range[1] * max_stages_per_job
        # stats for the Jobs
        self.job_stats: Dict[int, Dict[str, Any]] = {} # in-progress job stats
        self.completed_jobs: List[Dict[str, Any]] = [] # completed job stats
        self.last_node_for_job: Dict[int, int] = {} # last assigned node per job
        self.handoff_latency = handoff_latency
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
        If use_task_dependence is True, generates the first task of a new job, else we start with a single independent task.
        Returns initial state.
        """
        self.current_time = 0.0
        self.steps_done = 0
        self.next_job_id = 0
        for node in self.nodes:
            node.queue.clear()
            node.running_intervals.clear()
            node.busy_until = 0.0

        self.job_stats.clear()
        self.completed_jobs.clear()
        self.last_node_for_job.clear()
        if self.use_task_dependence:
            self.current_task = self._generate_job_first_task()
        else:
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
        prev_node_id = self.last_node_for_job.get(task.job_id) if self.use_task_dependence else None
        E_i, L_i, U_i, deadline_miss, overload, finish_time = self._simulate_task_on_node(task, chosen_node,prev_node_id=prev_node_id)
        if self.use_task_dependence:
            self.last_node_for_job[task.job_id] = chosen_node.node_id
        # update job stats if applicable
        if self.use_task_dependence:
            js = self.job_stats.get(task.job_id)
            if js is None:
                js = {
                    "first_arrival": task.arrival_time,
                    "deadline": task.deadline,
                    "num_stages": task.num_stages,
                    "energy": 0.0,
                    "finish_time": finish_time,
                    "stages_completed": 0,
                }
                self.job_stats[task.job_id] = js

            js["energy"] += E_i
            js["finish_time"] = max(js["finish_time"], finish_time)
            js["stages_completed"] += 1
            # If this was the last stage of the job, finalize job stats
            if task.stage_idx + 1 >= task.num_stages:
                job_latency = js["finish_time"] - js["first_arrival"]
                job_deadline_miss = js["finish_time"] > js["deadline"]
                self.completed_jobs.append(
                    {
                        "job_id": task.job_id,
                        "latency": job_latency,
                        "energy": js["energy"],
                        "deadline_miss": job_deadline_miss,
                        "num_stages": js["num_stages"],
                    }
                )
                # Remove job from active stats
                del self.job_stats[task.job_id]
                if task.job_id in self.last_node_for_job:
                    del self.last_node_for_job[task.job_id] # clean up last node record
                
        # compute reward
        reward = self._compute_reward(
            E_i=E_i,
            L_i=L_i,
            D_i=task.deadline,
            U_i=U_i,
            deadline_miss=deadline_miss,
            overload=overload,
            arrival=task.arrival_time,
        )
        # accounting for episode progress
        self.steps_done += 1
        done = self.steps_done >= self.episode_length
        next_state = None
        if not done:
            # advance time between arrivals (jobs/stages)
            delta_t = float(self._rng.uniform(*self.interarrival_range))
            self.current_time = max(self.current_time, task.arrival_time) + delta_t
            if self.use_task_dependence:
                if task.stage_idx + 1 < task.num_stages:
                    # there is a next stage in the same job
                    next_stage_idx = task.stage_idx + 1
                    # you can differentiate stage lengths if you want
                    length2 = float(self._rng.uniform(*self.task_length_range))
                    self.current_task = Task(
                        length_mi=length2,
                        deadline=task.deadline,           # same end-to-end deadline
                        arrival_time=self.current_time,   # arrives after previous stage
                        job_id=task.job_id,
                        stage_idx=next_stage_idx,
                        num_stages=task.num_stages,
                    )
                else:
                    # last stage of this job -> start a completely new job
                    self.current_task = self._generate_job_first_task()
            else:
                self.current_task = self._generate_task()

            next_state = self._build_state()
        else:
            self.current_task = None
        info = {
            "E_i": E_i,
            "L_i": L_i,
            "U_i": U_i,
            "deadline_miss": deadline_miss,
            "overload": overload,
            "chosen_node": chosen_node.name,
            # extra info for dependency experiments
            "job_id": task.job_id,
            "stage_idx": task.stage_idx,
            "num_stages": task.num_stages,
            "finish_time": finish_time,
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
        
    def _generate_job_first_task(self) -> Task:
        """
        Generate the first task of a new job with multiple stages.
        The job has a random number of stages between 2 and max_stages_per_job.
        All stages share the same end-to-end deadline 
        """
        length = float(self._rng.uniform(*self.task_length_range))
        slack = float(self._rng.uniform(*self.deadline_slack_range))
        arrival = self.current_time  # one task per step at current time
        deadline = arrival + slack
        job_id = self.next_job_id
        self.next_job_id += 1
        num_stages = self._rng.integers(2, self.max_stages_per_job + 1) # random number of stages per job, at least 2 stages

        return Task(
            length_mi=length,
            deadline=deadline,
            arrival_time=arrival,
            job_id=job_id,
            stage_idx=0,
            num_stages=num_stages,
        )
        
    def _simulate_task_on_node(
    self,
    task: Task,
    node: Node,
    prev_node_id: Optional[int] = None,
) -> Tuple[float, float, float, bool, bool, float]:
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
        handoff = 0.0
        if prev_node_id is not None and prev_node_id != node.node_id:
            handoff = self.handoff_latency
        completion_time = end_time + net_delay + handoff
        L_i = (completion_time - task.arrival_time)
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
        return E_i, L_i, U_i, deadline_miss, overload, end_time

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
        Build the observation vector.
        Features:
        - Node features: time until each node becomes free (normalized)
        - Task features:
            * Normalized length
            * Normalized slack
            * Stage progress (0=start, 1=end)
            * Remaining stages (normalized)
            * Time to deadline normalized
        """
        node_feats = []
        # normalisation factor for busy_until differences
        max_future = max(self.deadline_slack_range[1], 1.0)
        for node in self.nodes:
            avail_in = max(0.0, node.busy_until - self.current_time)
            node_feats.append(avail_in / max_future)
        t = self.current_task
        if t is None:
            # return zero vector if no task
            return np.zeros(len(node_feats) + 5, dtype=np.float32)

        # Normalized task length
        max_len = float(self.task_length_range[1])
        length_norm = min(1.0, t.length_mi / max_len)
        # Normalized slack (per-stage view)
        slack = max(0.0, t.deadline - self.current_time)
        max_slack = float(self.deadline_slack_range[1])
        slack_norm = min(1.0, slack / max_slack)
        # Stage progress (job-aware)
        #  0.0 --> first stage, 1.0 --> last stage
        if t.num_stages > 1:
            stage_progress = t.stage_idx / float(t.num_stages - 1)
        else:
            stage_progress = 0.0
        # Remaining stages (normalized)
        remaining_stages = (t.num_stages - t.stage_idx) / float(t.num_stages)
        # Time to full job deadline (normalized)
        # use a reference so that max possible slack over all stages
        time_to_deadline = slack
        deadline_ref = max(1e-6, self.deadline_slack_range[1] * self.max_stages_per_job)
        time_to_deadline_norm = min(1.0, time_to_deadline / deadline_ref)

        task_feats = [
            length_norm,
            slack_norm,
            stage_progress,
            remaining_stages,
            time_to_deadline_norm,
        ]

        state = np.array(node_feats + task_feats, dtype=np.float32)
        return state