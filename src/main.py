"""
    File: main.py
    Author: Gabriel Biel

    Description: Main file to train and test the agents. It loads the configuration,
    initializes the environment, and the agent. It trains the agent and saves the trained
    model and logs the training summary. It also tests the agent and logs the testing summary.
"""
import os
import torch
import time

from algorithms.tools.logger import Logger
from tools.config_loader import load_config, config_to_yaml
from tools.arguments_parser import get_args

from env.environment import TempRegulationEnv
from algorithms.network import Network as Network
from algorithms.ppo.agent import Agent as PPOAgent
from algorithms.dql.agent import Agent as DQLAgent

def train_ppo(agent:PPOAgent, actor_model, critic_model, total_timesteps=None):
    if total_timesteps is None:
        total_timesteps = 4*24*360*15
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

def test_ppo(agent:PPOAgent, actor_model, total_timesteps=None):
    if total_timesteps is None:
        total_timesteps = 4*24*14
    print(f"Testing PPO {actor_model}", flush=True)
    try:
        print(f"Loading actor model.", flush=True)
        agent.load_actor(actor_model)
    except FileNotFoundError:
        print(f"Could not find ${actor_model}. Please provide a valid path to actor model to test.", flush=True)
        return
    agent.test_policy(total_timesteps)

def train_dql(agent:DQLAgent, local_qnetwork, target_qnetwork, total_timesteps=None):
    if total_timesteps is None:
        total_timesteps = 4*24*360*15
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

def test_dql(agent:DQLAgent, local_qnetwork, total_timesteps=None):
    if total_timesteps is None:
        total_timesteps = 4*24*14
    print(f"Testing DQL {local_qnetwork} target Q network.", flush=True)
    try:
        agent.local_qnetwork.load_state_dict(torch.load(local_qnetwork))

    except FileNotFoundError:
        print(f"Could not find {local_qnetwork}. Please provide a valid path to local Q network to test.", flush=True)
        return
    agent.test_policy(total_timesteps)

def load_agent(env, agent_type, config):
    NETWORK = config['network']
    HYPERPARAMETERS = config['hyperparameters']
    if agent_type.upper() == 'PPO':
        actor = Network(
            env.observation_space.shape[0],
            env.action_space.n,
            NETWORK['hidden_layers'],
            NETWORK['activation'],
            NETWORK['output_activation'])
        critic = Network(
            env.observation_space.shape[0],
            1,
            NETWORK['hidden_layers'],
            NETWORK['activation'],
            NETWORK['output_activation'])
        agent = PPOAgent(env=env, actor_network=actor, critic_network=critic, **HYPERPARAMETERS)
    elif agent_type.upper() == 'DQL':
        local_qnetwork = Network(
            env.observation_space.shape[0],
            env.action_space.n,
            NETWORK['hidden_layers'],
            NETWORK['activation'],
            NETWORK['output_activation'])
        target_qnetwork = Network(
            env.observation_space.shape[0],
            env.action_space.n,
            NETWORK['hidden_layers'],
            NETWORK['activation'],
            NETWORK['output_activation'])
        agent = DQLAgent(env=env, local_qnetwork=local_qnetwork, target_qnetwork=target_qnetwork, **HYPERPARAMETERS)
    else:
        print(f"Unknown agent type '{agent_type}'. Exiting.", flush=True)
        return None
    return agent

def print_training_summary(agent, folder_path, elapsed_time, CONFIG):
    print('-------Training completed-------')
    print(f"Training took {int(elapsed_time//60)} minutes and {elapsed_time%60:.2f} seconds.")
    Logger().save_agent_info(folder_path, agent, CONFIG, elapsed_time)
    Logger().save_trained_agent(agent, folder_path)
    Logger().plot_scores(folder_path)
    Logger().save_all_aio_plots(agent, folder_path)
    print(f'Trained model and summary available in {folder_path} folder.')
    print('--------------------------------')
    agent.env.close()

def print_start_message(agent, agent_type, mode, total_timesteps):
    print('--------------------------------')
    print(f'Starting {agent_type} {mode}ing.')
    print(f'Total timesteps: {total_timesteps}')
    print(f'This is an equivalent of {total_timesteps//(4*24)} days - {total_timesteps//agent.env.max_steps_per_episode} environment episodes.')
    print('--------------------------------')

def main():
    args = get_args()

    CONFIG = load_config(args.config, args.algorithm)

    env = TempRegulationEnv(
        start_from_random_day=True,
        seed=int(args.seed),
        max_steps_per_episode=4*24*14,
    )

    start_time = time.time()
    agent = load_agent(env, args.algorithm, CONFIG)
    if agent is None:
        print(f"Could not load agent. Exiting.", flush=True)
        return

    print_start_message(agent, args.algorithm, args.mode, args.total_timesteps)

    try:
        if args.algorithm == 'PPO':
            if args.mode == 'train':
                train_ppo(agent, args.actor_model, args.critic_model, total_timesteps=args.total_timesteps)
            else:
                test_ppo(agent, args.actor_model, total_timesteps=args.total_timesteps)

        elif args.algorithm == 'DQL':
            if args.mode == 'train':
                train_dql(agent, args.local_qnetwork, args.target_qnetwork, total_timesteps=args.total_timesteps)
            elif args.mode == 'test':
                test_dql(agent, args.local_qnetwork, total_timesteps=args.total_timesteps)

    except KeyboardInterrupt:
        print(f'\n{args.algorithm} {args.mode}ing interrupted by user.', flush=True)

    finally:
        # Save the agent and plot the average score overtime
        if args.mode == 'train':
            elapsed_time = time.time() - start_time
            folder_path = f"out/{args.algorithm.lower()}/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(folder_path, exist_ok=True)
            # Copy config file to the output folder
            config_to_yaml(CONFIG, f"{folder_path}/config.yaml")
            print_training_summary(agent, folder_path, elapsed_time, CONFIG)
        if args.mode == 'test':
            folder_path = f"out/testing/{args.algorithm.lower()}/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(folder_path, exist_ok=True)
            Logger().save_all_aio_plots(agent, folder_path)

if __name__ == '__main__':
    main()
