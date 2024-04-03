import argparse
import torch

from env.environment import TempRegulationEnv
from algorithms.tools.logger import Logger

# PPO imports
from algorithms.ppo.agent import Agent as PPOAgent
from algorithms.ppo.network import Network as PPONetwork
from algorithms.ppo.eval_policy import eval_policy as eval_ppo_policy

# DQL imports
from algorithms.dql.agent import Agent as DQLAgent
from algorithms.dql.network import Network as DQLNetwork

def train_ppo(env, hyperparameters, actor_model, critic_model):
    print(f"Training PPO", flush=True)
    agent = PPOAgent(policy_class=PPONetwork, env=env, **hyperparameters)
    if actor_model != '' and critic_model != '':
        agent.actor.load_state_dict(torch.load(actor_model))
        agent.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training from scratch.", flush=True)
    agent.train(total_timesteps=200_000_000)

def test_ppo(env, actor_model):
    print(f"Testing PPO {actor_model}", flush=True)
    agent = PPOAgent(policy_class=PPONetwork, env=env)
    agent.actor.load_state_dict(torch.load(actor_model))
    agent.test_policy()

def train_dql(env, hyperparameters):
    print('Training DQL', flush=True)
    agent = DQLAgent(env=env, policy_class=DQLNetwork, **hyperparameters)
    try:
        agent.train(hyperparameters['number_episodes'])
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current agent...")
    finally:
        Logger().save_agent(agent)
        print("Agent saved successfully.")
        env.close()
        Logger().plot_scores()
        print("Scores plotted successfully.")

def test_dql(env, actor_model):
    print(f"Testing DQL {actor_model}", flush=True)
    agent = DQLAgent(env=env, policy_class=DQLNetwork)
    agent.policy.load_state_dict(torch.load(actor_model))
    agent.test_policy()

def main():
    parser = argparse.ArgumentParser(description='Train or test PPO/DQL model.')
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=False, default='train' , help='Mode to run the script in')
    parser.add_argument('-a', '--algorithm', choices=['PPO', 'DQL'], required=False, default='DQL', help='Algorithm to use')
    parser.add_argument('--actor_model', required=False, default='', help='Path to the actor model file')
    parser.add_argument('--critic_model', required=False, default='', help='Path to the critic model file - only for PPO')
    args = parser.parse_args()

    env = TempRegulationEnv()

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
        if args.mode == 'train':
            train_ppo(env, hyperparameters, args.actor_model, args.critic_model)
        else:
            test_ppo(env, args.actor_model)
    elif args.algorithm == 'DQL':
        hyperparameters = {
            'number_episodes': 1500,
            'epsilon_starting_value': 1.0,
            'epsilon_ending_value': 0.01,
            'epsilon_decay_value': 0.995,
            'learning_rate': 5e-4,              # Learning rate for the optimizer
            'minibatch_size': 100,              # Size of the minibatch from replay memory for learning
            'discount_factor': 0.99,            # Discount factor for future rewards
            'replay_buffer_size': int(1e5),     # Size of the replay buffer
            'interpolation_parameter': 1e-3     # Used in soft update of target network
        }
        if args.mode == 'train':
            train_dql(env, hyperparameters)
        elif args.mode == 'test':
            test_dql(env, args.actor_model)

if __name__ == '__main__':
    main()
