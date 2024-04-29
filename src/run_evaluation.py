import os
import time
import argparse
from algorithms.tools.logger import Logger
from env.environment import TempRegulationEnv
from main import load_agent
from tools.config_loader import load_config, config_to_yaml

def print_line():
    print("========================================", flush=True)

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate PPO/DQL model.')
    parser.add_argument('-a', '--algorithm', choices=['PPO', 'DQL', 'ppo', 'dql'], required=False, default='dql', type=str, help='Algorithm to use')
    parser.add_argument('--config', required=False, default='', type=str, help='Path to the config file. Specifies hyperparameters and network parameters for algorithm')
    parser.add_argument('--output', required=False, default='out/results/eval', type=str, help='Output directory for results')
    return parser.parse_args()

def save_agent_info(folder_path, agent, config, elapsed_time, extra_text=""):
    os.makedirs(folder_path, exist_ok=True)
    Logger().save_agent_info(f"{folder_path}", agent, config, elapsed_time, extra_text)
    Logger().save_trained_agent(agent, folder_path)
    Logger().plot_scores(f"{folder_path}")
    Logger().plot_all_in_one(agent, f"{folder_path}")
    config_to_yaml(config, f"{folder_path}/config.yaml")

def get_enviroment(seed):
    return TempRegulationEnv(
        start_from_random_day=True,
        seed=int(seed),
        max_steps_per_episode=4*24*14,
    )

def reset_learning(env, agent, seed, algorithm, config):
    Logger().reset()
    if env:
        del env
    if agent:
        del agent
    env = get_enviroment(seed)
    agent = load_agent(env, algorithm, config)
    return env, agent

def main():
    args = get_args()
    algorithm = args.algorithm.lower()
    os.makedirs(args.output, exist_ok=True)

    print(f"Running evaluation for {algorithm.upper()}", flush=True)
    seeds = [1, 2, 3, 4, 5]
    year = 4 * 24 * 360
    learning_lengths = [25, 50, 100]
    results = {}
    env = None
    agent = None

    config = load_config(args.config, args.algorithm, silent=True)

    try:
        for seed in seeds:
            results[seed] = {}
            print_line()
            print(f"Training in environment {seed}.", flush=True)
            for learning_length in learning_lengths:
                print(f"Env: {seed} : {learning_length} years.", flush=True)
                env, agent = reset_learning(env, agent, seed, algorithm, config)
                training_beginning = time.time()
                agent.train(learning_length * year)
                elapsed_time = time.time() - training_beginning
                print(f"Training time: {elapsed_time / 60} minutes.")
                results[seed][learning_length] = Logger().get_last_avg_score()

                summary  = f"""
Seed: {seed}, Learning length: {learning_length} years.
Last average score: {results[seed][learning_length]}
All score:
{Logger().all_scores}
                """

                save_agent_info(f"{args.output}/{seed}:{learning_length}", agent, config, elapsed_time, extra_text=summary)
    except KeyboardInterrupt:
        print("Interrupted by user, saving agent and results...")
    finally:
        print_line()
        print("Results:")
        for seed, seed_results in results.items():
            for learning_length, scores in seed_results.items():
                print(f"Seed: {seed}, Learning length: {learning_length} years. Last average score: {scores}")

        with open(f"{args.output}/results.txt", "w") as f:
            f.write("Results:\n")
            for seed, seed_results in results.items():
                for learning_length, scores in seed_results.items():
                    f.write(f"Seed: {seed}, Learning length: {learning_length} years. Last average score: {scores}\n")

if __name__ == '__main__':
    main()
