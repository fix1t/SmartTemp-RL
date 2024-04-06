import argparse
import torch
import time

from env.environment import TempRegulationEnv
from algorithms.tools.logger import Logger

from algorithms.network import Network as Network

# PPO imports
from algorithms.ppo.agent import Agent as PPOAgent

# DQL imports
from algorithms.dql.agent import Agent as DQLAgent

def train_ppo(agent:PPOAgent, actor_model, critic_model, total_timesteps=4*24*360*20):
    print(f"Training PPO", flush=True)
    if actor_model != '' and critic_model != '':
        try:
            print(f"Loading actor and critic models.", flush=True)
            agent.load_actor(actor_model)
            agent.load_critic(critic_model)
        except FileNotFoundError:
            print(f"Could not find {actor_model} or {critic_model}. Please provide a valid path to actor and critic models to load.", flush=True)
            return
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training from scratch.", flush=True)

    agent.train(total_timesteps)

def test_ppo(agent:PPOAgent, actor_model, total_timesteps=4*24*14):
    print(f"Testing PPO {actor_model}", flush=True)
    try:
        print(f"Loading actor model.", flush=True)
        agent.load_actor(actor_model)
    except FileNotFoundError:
        print(f"Could not find ${actor_model}. Please provide a valid path to actor model to test.", flush=True)
        return
    agent.test_policy(total_timesteps)

def train_dql(agent:DQLAgent, local_qnetwork, target_qnetwork, total_timesteps=4*24*360*20):
    print('Training DQL', flush=True)
    # TODO: Pass local and target Q networks

    if local_qnetwork != '' and target_qnetwork != '':
        try:
            print(f"Loading local and target Q networks.", flush=True)
            agent.load_local_qnetwork(local_qnetwork)
            agent.load_target_qnetwork(target_qnetwork)
        except FileNotFoundError:
            print(f"Could not find {local_qnetwork} or {target_qnetwork}. Please provide a valid path to local and target Q networks to load.", flush=True)
            return
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training from scratch.", flush=True)

    agent.train(total_timesteps)

def test_dql(agent:DQLAgent, local_qnetwork, total_timesteps=4*24*14):
    print(f"Testing DQL {local_qnetwork} target Q network.", flush=True)
    try:
        agent.local_qnetwork.load_state_dict(torch.load(local_qnetwork))

    except FileNotFoundError:
        print(f"Could not find {local_qnetwork}. Please provide a valid path to local Q network to test.", flush=True)
        return
    agent.test_policy(total_timesteps)

def main():
    parser = argparse.ArgumentParser(description='Train or test PPO/DQL model.')
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=False, default='train' , type=str ,help='Mode to run the script in. Test mode requires a model file.')
    parser.add_argument('-a', '--algorithm', choices=['PPO', 'DQL'], required=False, default='DQL', type=str ,help='Algorithm to use')
    parser.add_argument('--actor_model', required=False, default='', type=str ,help='Path to the actor model file - only for PPO')
    parser.add_argument('--critic_model', required=False, default='', type=str ,help='Path to the critic model file - only for PPO')
    parser.add_argument('--local_qnetwork', required=False, default='', type=str ,help='Path to the local qnetwork model file - only for DQL')
    parser.add_argument('--target_qnetwork', required=False, default='', type=str ,help='Path to the target qnetwork model file - only for DQL')
    parser.add_argument('--seed', required=False, default='',  type=int, help='Seed for the environment')
    args = parser.parse_args()

    if args.seed == '':
        args.seed = None

    env = TempRegulationEnv(
        start_from_random_day=True,
        seed=int(args.seed),
        max_steps_per_episode=4*24*14,
    )

    try:
        start_time = time.time()
        if args.algorithm == 'PPO':
            hyperparameters = {
                'timesteps_per_batch': 2048,
                'max_timesteps_per_episode': 200,
                'gamma': 0.99,
                'n_updates_per_iteration': 10,
                'lr': 3e-4,
                'clip': 0.2,
                'render': False,
                'render_every_i': 10
            }

            # Network for PPO
            actor = Network(env.observation_space.shape[0], env.action_space.n, [64, 64])
            critic = Network(env.observation_space.shape[0], 1,[64, 64])
            agent = PPOAgent(actor_network=actor, critic_network=critic, env=env, **hyperparameters)

            if args.mode == 'train':
                train_ppo(agent, args.actor_model, args.critic_model)
            else:
                test_ppo(agent, args.actor_model)

        elif args.algorithm == 'DQL':
            hyperparameters = {
                'epsilon_starting_value': 1.0,
                'epsilon_ending_value': 0.01,
                'epsilon_decay_value': 0.995,
                'learning_rate': 5e-4,              # Learning rate for the optimizer
                'minibatch_size': 100,              # Size of the minibatch from replay memory for learning
                'discount_factor': 0.99,            # Discount factor for future rewards
                'replay_buffer_size': int(1e5),     # Size of the replay buffer
                'interpolation_parameter': 1e-3     # Used in soft update of target network
            }

            # Network for DQL
            local_qnetwork = Network(env.observation_space.shape[0], env.action_space.n, [64, 64])
            target_qnetwork = Network(env.observation_space.shape[0], env.action_space.n, [64, 64])
            agent = DQLAgent(env=env, local_qnetwork=local_qnetwork, target_qnetwork=target_qnetwork, **hyperparameters)

            if args.mode == 'train':
                train_dql(agent, args.local_qnetwork, args.target_qnetwork)
            elif args.mode == 'test':
                test_dql(agent, args.local_qnetwork)

    except KeyboardInterrupt:
        print(f'\n{args.algorithm} {args.mode}ing interrupted by user.', flush=True)

    finally:
        # Save the agent and plot the average score overtime
        if args.mode == 'train':
            elapsed_time = time.time() - start_time
            print('-------Training completed-------')
            print("Saving agent and plotting scores...")
            save_folder = f"out/{args.algorithm}"
            Logger().save_agent(agent, save_folder)
            print("Agent saved successfully.")
            Logger().plot_scores(save_folder)
            print("Scores plotted successfully.")
            print(f'Trained model available in {save_folder} folder.')
            print(f"Training took {elapsed_time/60:.2f} minutes and {elapsed_time%60:.2f} seconds.")
            print('--------------------------------')
            env.close()

if __name__ == '__main__':
    main()
