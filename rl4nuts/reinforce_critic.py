import os
import time
import re

import random
import argparse
import logging
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from distutils.util import strtobool
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define the policy network, aka the actor, and the value network, aka the critic
class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()

        self.actor = nn.Sequential(
            # NOTE: obtain input and output dimensions from the environment
            layer_init(nn.Linear(env.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(env.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )


    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)  # probability distribution
        if action is None:
            action = probs.sample()         # pi(a|s) in action via pd
        return action, probs.log_prob(action), self.critic(x)


# Define a function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str,
                         default=os.path.basename(__file__).rstrip(".py"),
                        help='The name of this experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
                        help='The id of the gym environment to use')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='The total number of episodes to run')
    parser.add_argument('--eval-frequency', type=int, default=2000,
                        help='Every <eval-frequency> steps, the agent will be evaluated')
    parser.add_argument('--eval-episodes', type=int, default=8,
                        help='The number of episodes to run for each evaluation')
    parser.add_argument('--actor-learning-rate', type=float, default=2.5e-4,
                        help='the actor learning rate of the optimizer')
    parser.add_argument('--critic-learning-rate', type=float, default=3.5e-4,
                        help='the critic learning rate of the optimizer')
    parser.add_argument('--warmup-episodes', type=int, default=10,
                        help='the number of warmup episodes to run')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the gamma factor for compute the discounted return')
    parser.add_argument('--seed', type=int, default=666,
                        help='The random seed to use for the experiment')
    parser.add_argument('--total-timesteps', type=int, default=250000,
                        help='The total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)),
                         default=True, nargs='?', const=True,
                           help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                        default='mps', help='Device to perform tensor ops.')
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='if toggled, track the experiment with W&B')
    parser.add_argument('--wandb-project-name', type=str, default='rl4nuts',
                        help='The name of the W&B project to use')
    parser.add_argument('--wandb-entity', type=str, default="alcazar90",
                        help='The W&B entity (team) to use for the wandb project')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='Wether to capture video of the environment')

    # how much data we want to collect...
    parser.add_argument('--num-steps', type=int, default=200,
                        help='Number of steps to run for each environment per update')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
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

    logger.info(f"env.single_observation_space: {env.observation_space.shape}")
    logger.info(f"env.single_action_space.n: {env.action_space.n}")


    if args.capture_video:
        # Ref: https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="videos",
            episode_trigger=lambda t: t % 100 == 0,
            name_prefix=f"reinforce_critic-{run_name}",
            )
    # Ref: https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordEpisodeStatistics
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Check some environment information
    logger.info(f"Max episode steps: {env.spec.max_episode_steps}")

    # Initialize the agent and the optimizer
    agent = Agent(env).to(device)
    # NOTE: use PPO's epsilon by default (1e-5) instead of PyTorch's default (1e-8)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.actor_learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.critic_learning_rate, eps=1e-5)


    observation = env.reset(seed=args.seed)[0]

    logger.info("observation: %s", observation)
    logger.info("observation type: %s", type(observation))

    # Buffers trajectory information
    obs = torch.zeros((args.num_steps, *env.observation_space.shape)).to(device)
    actions = torch.zeros((args.num_steps,), dtype=torch.long).to(device)
    values = torch.zeros(args.num_steps).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(observation).to(device)  # initial observation
    next_done = torch.zeros(1).to(device)  # initial done state

    # actions = torch.zeros((args.num_steps, env.action_space.n)).to(device)
    for i in range(1, args.num_episodes + 1):
        episode_start_time = time.time()

        # Inner loop = collect a trajectory, aka policy roll out
        for step in range(0, args.num_steps):
            global_step += 1

            obs[step] = next_obs
            dones[step] = next_done

            action, logprob, value = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.squeeze()

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

                logger.info("[episode %s] episodic_return: %s | episodic_length: %s | episodic_time: %s", i, episodic_return, episodic_length, episodic_time)

                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                writer.add_scalar("charts/episodic_time", episodic_time, global_step)

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

        # define which baseline to use
        if i <= args.warmup_episodes:
            advantages = discounted_returns - discounted_returns.mean()
            advantages = advantages - advantages.mean()
            logger.info(f"[WARMUP] Episode {i}: Using centered baseline")
        else:
            # After warmpup: use critic as baseline
            advantages = (discounted_returns - values[:episodic_length]).detach()
            advantages = advantages - advantages.mean()
            logger.info(f"[LEARNING] Episode {i}: Using critic as baseline")

        advantages = torch.clamp(advantages, -10, 10)  # Prevent extreme values

        # Compute the actor loss and update the actor network
        actor_loss = - (logprobs[:episodic_length] * advantages).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=0.5)
        actor_optimizer.step()


        # Compute the critic loss and update the critic network
        value_estimates = values[:episodic_length]
        # returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
        critic_loss = nn.MSELoss()(value_estimates, returns)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)
        critic_optimizer.step()

        logger.info(f"[episode {i}] returns: mean={discounted_returns.mean():.2f}, std={discounted_returns.std():.2f}")
        logger.info(f"[episode {i}] values: mean={values[:episodic_length].mean():.2f}, std={values[:episodic_length].std():.2f}")
        logger.info(f"[episode {i}] advantages: mean={advantages.mean():.2f}, std={advantages.std():.2f}, max={advantages.max():.2f}, min={advantages.min():.2f}")

        logger.info("[episode %s] discounted return at t=0: %s | return: %s | actor_loss: %s | critic_loss: %s | episodic_length: %s", i, discounted_returns[0].item(), discounted_returns.mean().item(), actor_loss.item(), critic_loss.item(), episodic_length)

        # Detach all buffers to break gradient computation graph
        obs = obs.detach()
        actions = actions.detach()
        logprobs = logprobs.detach()
        rewards = rewards.detach()
        dones = dones.detach()
        returns = returns.detach()  # if you're reusing this
        values = values.detach()

        # Log relevant information
        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Evaluate the agent's performance every eval_frequency steps
        # NOTE: check if there is common to use a separate env for evaluation
        if i % 50 == 0 or global_step % args.eval_frequency == 0:
            print("Evaluating the agent...at step:", global_step)
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
                        eval_action, _, _ = agent.get_action_and_value(eval_obs_tensor)
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
            logger.info(f"Evaluation over {args.eval_episodes} episodes: Average Return: {avg_eval_return}, Average Length: {avg_eval_length}")
            writer.add_scalar("charts/eval_average_return", avg_eval_return, global_step)
            writer.add_scalar("charts/eval_average_length", avg_eval_length, global_step)

    # Check whether to log video the video or not in WandB
    if args.track and args.capture_video:
        print("Logging videos to W&B...")
        for video in sorted(os.listdir("videos"), key=lambda x: int(re.search(r'episode-(\d+)', x).group(1)) if re.search(r'episode-(\d+)', x) else 0):
            if video.endswith(".mp4"):
                print(f"Logging video {video} to wandb")
                wandb.log({"video": wandb.Video(os.path.join("videos", video), format="mp4")})

    env.close()
