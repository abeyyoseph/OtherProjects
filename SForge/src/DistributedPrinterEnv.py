import gymnasium as gym
from gym import spaces
import numpy as np
import sys
from Printer import Printer, PrinterState
from Job import Job

SHIFT_OVER = 480  # 8-hour shift in minutes

class DistributedPrinterEnv(gym.Env):
    def __init__(self, num_printers=3, total_jobs=10):
        self.num_printers = num_printers
        self.total_jobs = total_jobs
        self.current_min = 0
        self.step_duration = 2

        # reward weights
        self.alpha1 = 1.0
        self.alpha2 = 0.5
        self.alpha3 = 2.0

        # gym spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_printers * 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_printers)

        # instantiate printers and jobs
        self.printers = [Printer(i) for i in range(num_printers)]
        self.jobs = [Job(np.random.randint(10, 60), np.random.randint(50, 200)) for _ in range(total_jobs)]
        self.pending_jobs = list(self.jobs)

        self.completed_jobs_this_step = []
        self.failed_jobs_this_step = []
        self.total_completed_jobs = 0
        self.assignments_to_failed_printers = []

        print(f"[INIT] Environment created with {self.num_printers} printers and {self.total_jobs} jobs.")

    def reset(self):
        print("\n[RESET] Resetting environment...")
        self.current_min = 0
        for p in self.printers:
            p.reset()
        self.pending_jobs = list(self.jobs)
        self.total_completed_jobs = 0
        print(f"[RESET] Environment reset complete. {len(self.pending_jobs)} jobs ready.")
        return self._get_observation()

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
        throughput_gain = len(self.completed_jobs_this_step) / max(1, self.total_jobs)
        idle_time = sum(p.state == PrinterState.IDLE for p in self.printers) / len(self.printers)
        failure_penalty = (
            len(self.failed_jobs_this_step) / max(1, self.total_jobs) +
            len(self.assignments_to_failed_printers) / max(1, self.num_printers)
        )
        reward = (self.alpha1 * throughput_gain - self.alpha2 * idle_time - self.alpha3 * failure_penalty)
        reward_clipped = np.clip(reward, -1.0, 1.0)
        print(f"[REWARD] Throughput={throughput_gain:.3f}, Idle={idle_time:.3f}, FailurePenalty={failure_penalty:.3f}, Reward={reward_clipped:.3f}")
        return reward_clipped

    def _check_done(self):
        done = self.current_min >= SHIFT_OVER or self.total_completed_jobs >= self.total_jobs
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
    if len(sys.argv) != 3:
        print("Usage: python3 DistributedPrinterEnv.py <num_printers> <num_jobs>")
        exit()

    desired_num_printers = int(sys.argv[1])
    desired_num_jobs = int(sys.argv[2])

    env = DistributedPrinterEnv(desired_num_printers, desired_num_jobs)
    env.execute_simulation()


    
