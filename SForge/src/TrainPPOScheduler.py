import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from DistributedPrinterEnv import DistributedPrinterEnv

# Instantiate the environment
env = DistributedPrinterEnv()

# Create PPO model
model = PPO(
    "MlpPolicy",      # simple feedforward policy
    env,              # environment
    verbose=1,        # print training info
    learning_rate=3e-4,
    n_steps=2048,     # Run environment for 2048 steps, compute gradients and update policy, then continue
    batch_size=64,    # After gathering n_steps of experience, PPO splits it into mini-batches for gradient optimization
    gamma=0.99, # It controls how far the agent “looks into the future.” Smaller value means more short-sighted (immediate reward)
)

# Train the model, each timestep corresponds to one env.step call in the DistributedPrinterEnv. The env will be reset when the
# "done" condition is met. 50,000 timesteps ≈ 208 simulated 8 hour shifts
model.learn(total_timesteps=50_000) 

# Save the model
model.save("ppo_distributed_scheduler")
env.close()
