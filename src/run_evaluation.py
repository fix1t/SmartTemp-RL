import os
import time
import argparse
from algorithms.tools.logger import Logger
from env.environment import TempRegulationEnv
from main import load_agent
from tools.config_loader import load_config, config_to_yaml

def print_divider():
    """Prints a line divider to the console."""
    print("=" * 40, flush=True)

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate PPO/DQL model.')
    parser.add_argument('-a', '--algorithm', choices=['PPO', 'DQL'], default='DQL', type=str.upper, help='Algorithm to use')
    parser.add_argument('--config', default='', type=str, help='Path to the config file with hyperparameters and network parameters')
    parser.add_argument('--output', default='out/results/eval', type=str, help='Output directory for results')
    return parser.parse_args()

def setup_environment(seed):
    """Initializes the temperature regulation environment with the given seed."""
    return TempRegulationEnv(start_from_random_day=True, seed=int(seed), max_steps_per_episode=4*24*14)

def reset_and_load_environment(env, agent, seed, algorithm, config):
    """Resets the learning environment and agent for a new training session."""
    Logger().reset()
    if env: del env
    if agent: del agent
    env = setup_environment(seed)
    agent = load_agent(env, algorithm, config)
    return env, agent

def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    print(f"Running evaluation for {args.algorithm}", flush=True)

    seeds = [1, 2, 3, 4, 5]
    year_duration_in_timesteps = 4 * 24 * 30
    checkpoint_every_n_steps = 1 * year_duration_in_timesteps
    total_learning_duration = 3 * year_duration_in_timesteps

    results = {}
    env, agent = None, None

    try:
        for seed in seeds:
            results[seed] = {}
            print_divider()
            print(f"Training in environment with seed {seed}.", flush=True)
            config = load_config(args.config, args.algorithm, silent=True)
            env, agent = reset_and_load_environment(env, agent, seed, args.algorithm, config)
            training_start_time = time.time()
            duration = 0
            checkpoint_num = 0

            while duration < total_learning_duration:
                checkpoint_num += 1
                duration += checkpoint_every_n_steps
                duration_years = duration // year_duration_in_timesteps
                print(f"Seed: {seed}, Duration: {duration_years} years.", flush=True)

                # Train the agent for the specified duration
                agent.train(min(total_learning_duration - duration, checkpoint_every_n_steps))
                training_duration = time.time() - training_start_time

                print(f"Training time: {training_duration / 60:.2f} minutes.")
                results[seed][checkpoint_num] = Logger().get_last_avg_score()
                summary = f"Seed: {seed}, Duration: {duration_years} years.\nLast average score: {results[seed][checkpoint_num]}\nAll scores:\n{Logger().all_scores}"
                save_agent_info(os.path.join(args.output, f"{seed}:{duration_years}"), agent, config, training_duration, summary)
    except KeyboardInterrupt:
        print("Interrupted by user, saving agent and results...")
    finally:
        print_divider()
        print("Results:")
        for seed, seed_results in results.items():
            for duration, scores in seed_results.items():
                print(f"Seed: {seed}, Duration: {duration} years. Last average score: {scores}")

        with open(os.path.join(args.output, "results.txt"), "w") as f:
            f.write("Results:\n")
            for seed, seed_results in results.items():
                for duration, scores in seed_results.items():
                    f.write(f"Seed: {seed}, Duration: {duration} years. Last average score: {scores}\n")

def save_agent_info(folder_path, agent, config, elapsed_time, extra_text=""):
    """Saves agent training info and configuration to a specified directory."""
    os.makedirs(folder_path, exist_ok=True)
    Logger().save_agent_info(folder_path, agent, config, elapsed_time, extra_text)
    Logger().save_trained_agent(agent, folder_path)
    Logger().plot_scores(folder_path)
    Logger().plot_all_in_one(agent, folder_path)
    config_to_yaml(config, os.path.join(folder_path, "config.yaml"))

if __name__ == '__main__':
    main()
