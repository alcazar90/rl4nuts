import os
import time
import re

import random
import logging
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from dataclasses import dataclass
from typing import Literal

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logger = logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Define the policy network, aka the agent
class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()

        self.actor = nn.Sequential(
            # NOTE: obtain input and output dimensions from the environment
            nn.Linear(env.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)  # probability distribution
        if action is None:
            action = probs.sample()         # pi(a|s) in action via pd
        # return action, probs.log_prob(action), probs.entropy()
        return action, probs.log_prob(action)


@dataclass
class Args:
    """Experiment Setting and Hyperparameters for REINFORCE algorithm."""

    exp_name: str = os.path.basename(__file__).rstrip(".py")
    """The name of the experiment."""

    gym_id: str = 'CartPole-v1'
    """The id of the gym environment to use."""

    num_episodes: int = 1000
    """The total number of episodes to run."""

    eval_frequency: int = 100
    """Evaluate every N episodes the agent's performance as average return over args.eval_episodes."""

    eval_episodes: int = 8
    """The number of episodes to run for each evaluation."""

    learning_rate: float = 2.5e-4
    """The learning rate of the optimizer."""

    gamma: float = 0.99
    """The gamma factor for compute the discounted return."""

    centered_returns: bool = False
    """If toggled, use centered returns, a simple baseline method to reduce variance."""

    seed: int = 666
    """The random seed to use for the experiment."""

    total_timesteps: int = 250000
    """The total timesteps of the experiments."""

    torch_deterministic: bool = True
    """If toggled, `torch.backends.cudnn.deterministic=False`."""

    device: Literal["mps", "cpu", "cuda"] = 'mps'
    """Device to perform tensor ops."""

    track: bool = False
    """If toggled, track the experiment with W&B."""

    wandb_project_name: str = 'rl4nuts'
    """The name of the W&B project to use."""

    wandb_entity: str = "alcazar90"
    """The W&B entity (team) to use for the wandb project."""

    capture_video: bool = False
    """Whether to capture video of the environment."""

    record_video_every_n_episodes: int = 100
    """Record video every n episodes."""

    num_steps: int = 500
    """Number of steps to run for each environment per update."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        # There is an issue with monitor_gym=True between wandb and gym
        # Ref: https://github.com/wandb/wandb/issues/10339
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        logger.info(f"Tracking experiment with W&B: {run_name}")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)

    if device.type == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available. Please check your PyTorch installation.")
        logger.info("Using Apple Silicon MPS device for tensor operations.")
    elif device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
        logger.info("Using CUDA device for tensor operations.")
    else:
        logger.info("Using CPU for tensor operations.")

    # env setup
    env = gym.make(args.gym_id, render_mode="rgb_array")
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if args.capture_video:
        # Ref: https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo
        video_folder = f"videos/{run_name}"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda t: t % args.record_video_every_n_episodes == 0,
            name_prefix=f"reinforce-{run_name}",
            )

    # Ref: https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordEpisodeStatistics
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Initialize the agent and the optimizer
    agent = Agent(env).to(device)
    # NOTE: use PPO's epsilon by default (1e-5) instead of PyTorch's default (1e-8)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observation = env.reset(seed=args.seed)[0]

    # Check some environment information
    logger.info(f"Environment: {args.gym_id}")
    logger.info(f"-> env.single_observation_space: {env.observation_space.shape}")
    logger.info(f"-> env.single_action_space.n: {env.action_space.n}")
    logger.info(f"-> max episode steps: {env.spec.max_episode_steps}")

    logger.info(f"-> observation {observation}")
    logger.info(f"-> observation type: {type(observation)}")

    # Buffers trajectory information
    obs = torch.zeros((args.num_steps, *env.observation_space.shape)).to(device)
    actions = torch.zeros((args.num_steps,), dtype=torch.long).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(observation).to(device)  # initial observation
    next_done = torch.zeros(1).to(device)  # initial done state

    # Track global_step at the end of each episode for video logging
    episode_to_step = {}

    # Outer loop = for each episode = experience collection + policy update
    for i in range(1, args.num_episodes + 1):
        episode_start_time = time.time()
        episodic_length = 0

        # Inner loop = collect an episode trajectory, aka policy roll out
        for step in range(0, args.num_steps):
            global_step += 1

            obs[step] = next_obs
            dones[step] = next_done

            action, logprob = agent.get_action(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            # perform environment step given the action
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.tensor(done).to(device)

            if done or step == args.num_steps - 1:
                # Note: info avoid self-record since episodic return
                # Ref: https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordEpisodeStatistics
                episodic_return = info['episode']['r'] if 'episode' in info.keys() else rewards[:step+1].sum().item()
                episodic_length = info['episode']['l'] if 'episode' in info.keys() else step + 1
                episodic_time = info['episode']['t'] if 'episode' in info.keys() else time.time() - episode_start_time

                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                writer.add_scalar("charts/episodic_time", episodic_time, global_step)

                # Track episode number to global_step mapping for video logging
                # Note: episode count starts at 0 in RecordVideo wrapper
                episode_to_step[i - 1] = global_step

                # reset environment for next episode and break inner loop for the current episode
                next_obs = torch.Tensor(env.reset()[0]).to(device)
                next_done = torch.zeros(1).to(device)
                break


        # Now the outer loop consume a complete/truncate trajectory to update the parameters, i.e.
        # of the agent for learning from experience

        # compute discounted returns
        discounted_returns = torch.zeros(episodic_length).to(device)
        future_return = 0.0

        for t in reversed(range(episodic_length)):
            future_return = rewards[t] + args.gamma * future_return * (1 - dones[t])
            discounted_returns[t] = future_return

        # NOTE: simple baseline method here to reduce variance, i.e. centered returns
        if args.centered_returns:
            baseline_return = discounted_returns.mean()
            discounted_returns = discounted_returns - baseline_return

        # Compute the loss
        loss = - (logprobs[:episodic_length] * discounted_returns).mean()

        logger.info(f"[Ep {i:4d}] R={episodic_return:7.1f} | L={episodic_length:3d} | Loss={loss.item():7.4f} | G0={discounted_returns[0].item():7.2f} | T(s)={episodic_time:3.2f}")

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Detach logprobs to break gradient computation graph and iterate the next episode
        # Avoid RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).
        logprobs = logprobs.detach()

        # Log relevant information
        writer.add_scalar("losses/policy_loss", loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Evaluate the agent's performance every eval_frequency steps
        if i % args.eval_frequency == 0:
            eval_returns = []
            eval_lengths = []
            for _ in range(args.eval_episodes):
                eval_obs = env.reset()[0]
                eval_done = False
                eval_ep_return = 0
                eval_ep_length = 0
                while not eval_done:
                    eval_obs_tensor = torch.Tensor(eval_obs).to(device)
                    with torch.no_grad():
                        eval_action, _ = agent.get_action(eval_obs_tensor)
                    eval_obs, eval_reward, eval_terminated, eval_truncated, eval_info = env.step(eval_action.cpu().numpy())
                    eval_done = eval_terminated or eval_truncated
                    # NOTE: use env wrappers to get episodic return
                    if "episode" in eval_info.keys():
                        eval_ep_return = eval_info['episode']['r']
                        eval_ep_length = eval_info['episode']['l']
                eval_returns.append(eval_ep_return)
                eval_lengths.append(eval_ep_length)
            avg_eval_return = np.mean(eval_returns)
            avg_eval_length = np.mean(eval_lengths)

            logger.info(f"[Evaluation ({args.eval_episodes}ep)] Avg R={avg_eval_return:7.1f} | Avg L={avg_eval_length:7.1f}")
            writer.add_scalar("evals/average_return", avg_eval_return, global_step)
            writer.add_scalar("evals/average_length", avg_eval_length, global_step)

            # Reset environment after evaluation to continue training
            next_obs = torch.Tensor(env.reset()[0]).to(device)
            next_done = torch.zeros(1).to(device)

    # Check whether to log video the video or not in wandb
    if args.track and args.capture_video:
        print("Logging videos to W&B...")
        for video in sorted(os.listdir(video_folder), key=lambda x: int(re.search(r'episode-(\d+)', x).group(1)) if re.search(r'episode-(\d+)', x) else 0):
            if video.endswith(".mp4"):
                # Extract episode number from filename (e.g., "episode-50" -> 50)
                episode_match = re.search(r'episode-(\d+)', video)
                if episode_match:
                    episode_num = int(episode_match.group(1))
                    # Get the corresponding global_step for this episode
                    step = episode_to_step.get(episode_num, None)
                    if step is not None:
                        print(f"Logging video {video} to wandb at step {step}")
                        wandb.log({"video": wandb.Video(os.path.join(video_folder, video), format="mp4")}, step=step)
                    else:
                        # Log final video using the last global_step (final episode recorded on env.close())
                        print(f"Logging final video {video} to wandb at step {global_step}")
                        wandb.log({"video": wandb.Video(os.path.join(video_folder, video), format="mp4")}, step=global_step)

    env.close()
    writer.close()
