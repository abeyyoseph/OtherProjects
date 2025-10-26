import gymnasium as gym
from gym import spaces
import numpy as np
import sys
from Printer import Printer, PrinterState
from Job import Job

SHIFT_OVER = 480  # 8-hour shift in minutes

class DistributedPrinterEnv(gym.Env):
    def __init__(self, max_printers=10, max_jobs=15):
        self.max_printers = max_printers
        self.max_jobs = max_jobs

        # default placeholders (actual values randomized on reset)
        self.num_printers = 0
        self.num_jobs = 0
        self.current_min = 0
        self.step_duration = 2 # 2 mins pass every step

        # reward weights
        self.alpha1 = 1.0
        self.alpha2 = 0.5
        self.alpha3 = 2.0

        # temporary gym spaces — will be resized after first reset
        # start with minimal valid dimensions to satisfy Gym interface
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(max_printers * 3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(max_printers)

        # initialize empty structures
        self.printers = []
        self.jobs = []
        self.pending_jobs = []

        self.completed_jobs_this_step = []
        self.failed_jobs_this_step = []
        self.total_completed_jobs = 0
        self.assignments_to_failed_printers = []

        print("[INIT] Environment initialized — printers and jobs will be created on reset.")

    def reset(self):
        print("\n[RESET] Resetting environment...")

        # Randomize active environment parameters (but don’t redefine spaces)
        self.num_printers = np.random.randint(2, self.max_printers + 1)
        self.num_jobs = np.random.randint(2, self.max_jobs + 1)

        # Recreate printers
        self.printers = [Printer(i) for i in range(self.num_printers)]

        # Generate randomized jobs
        self.jobs = [
            Job(
                estimated_duration=np.random.randint(10, 100),     # duration 10–100
                required_material=np.random.randint(50, 300),      # material 50–300
                priority=np.random.randint(0, 3),                  # priority 0–2
                split_potential=np.random.choice([True, False])    # randomly splittable
            )
            for _ in range(self.num_jobs)
        ]

        # Reset dynamic state
        self.current_min = 0
        self.pending_jobs = list(self.jobs)
        self.total_completed_jobs = 0

        # Construct padded observation
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[:self.num_printers * 3] = np.random.random(self.num_printers * 3)

        # Logging
        print(f"[RESET] Environment randomized:")
        print(f"        Printers: {self.num_printers}")
        print(f"        Jobs: {self.num_jobs}")
        print(f"        Example job durations/materials: "
            f"{[j.estimated_duration for j in self.jobs[:3]]} / "
            f"{[j.required_material for j in self.jobs[:3]]}")
        print(f"[RESET] Environment reset complete. {len(self.pending_jobs)} jobs ready.")

        return obs, {}

    def step(self, action):
        print(f"\n[STEP] Current minute: {self.current_min}, Action taken: Assign to printer {action}")

        self.completed_jobs_this_step = []
        self.failed_jobs_this_step = []
        self.assignments_to_failed_printers = []

        # assign next job if available
        if self.pending_jobs:
            job = self.pending_jobs.pop(0)
            printer = self.printers[action]
            print(f"[ASSIGN] Assigning Job {job.job_id} to Printer {printer.id}")
            success = printer.assign_job(job)
            if not success:
                print(f"[ASSIGN FAIL] Printer {printer.id} could not accept Job {job.job_id}")
                self.assignments_to_failed_printers.append(job)
        else:
            print("[ASSIGN] No pending jobs left.")

        # update all printers
        for p in self.printers:
            prev_state = p.state
            p.update(self.step_duration)
            if p.completed_job:
                print(f"[UPDATE] Printer {p.id} completed a job.")
                self.completed_jobs_this_step.append(p.completed_job)
                p.completed_job = False
                self.total_completed_jobs += 1
            if p.failed_job:
                print(f"[UPDATE] Printer {p.id} experienced a failure.")
                self.failed_jobs_this_step.append(p.failed_job)
                p.failed_job = False
            if prev_state != p.state:
                print(f"[STATE CHANGE] Printer {p.id}: {prev_state.value} -> {p.state.value}")

        reward = self._compute_reward()
        done = self._check_done()
        obs = self._get_observation()
        info = {}

        print(f"[STEP END] Reward: {reward:.3f}, Done: {done}, Total Completed: {self.total_completed_jobs}")
        self.current_min += self.step_duration

        return obs, reward, done, info

    def _get_observation(self):
        obs = []
        for p in self.printers:
            state_val = {
                PrinterState.IDLE: 0.0,
                PrinterState.PRINTING: 0.5,
                PrinterState.REPAIRING: 0.75,
                PrinterState.FAILED: 1.0
            }[p.state]
            obs.extend([
                state_val,
                p.material_level / Printer.PRINTER_MATERIAL_FULL,
                1.0 if p.current_job else 0.0
            ])
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        throughput_gain = len(self.completed_jobs_this_step) / max(1, self.num_jobs)
        idle_time = sum(p.state == PrinterState.IDLE for p in self.printers) / len(self.printers)
        failure_penalty = (
            len(self.failed_jobs_this_step) / max(1, self.num_jobs) +
            len(self.assignments_to_failed_printers) / max(1, self.num_printers)
        )
        reward = (self.alpha1 * throughput_gain - self.alpha2 * idle_time - self.alpha3 * failure_penalty)
        reward_clipped = np.clip(reward, -1.0, 1.0)
        print(f"[REWARD] Throughput={throughput_gain:.3f}, Idle={idle_time:.3f}, FailurePenalty={failure_penalty:.3f}, Reward={reward_clipped:.3f}")
        return reward_clipped

    def _check_done(self):
        done = self.current_min >= SHIFT_OVER or self.total_completed_jobs >= self.num_jobs
        if done:
            print(f"[DONE] Simulation complete at {self.current_min} minutes. Total jobs completed: {self.total_completed_jobs}")
        return done

    def execute_simulation(self):
        print("\n[SIMULATION START]")
        obs = self.reset()
        done = False
        while not done:
            action = np.random.randint(0, self.num_printers)
            obs, reward, done, info = self.step(action)
        print("[SIMULATION END]")


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage: python3 DistributedPrinterEnv.py")
        exit()

    env = DistributedPrinterEnv()
    env.execute_simulation()


    
