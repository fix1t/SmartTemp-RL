import argparse
import os
import time
import signal
from algorithms.tools.logger import Logger
from tools.config_generator import generate_nn_configurations, generate_hp_configurations
from tools.config_loader import load_config
from env.environment import TempRegulationEnv
from main import load_agent
import tools.generate_latex_table as glt

def save_agent_info(folder_path, agent, config, elapsed_time, save):
    os.makedirs(folder_path, exist_ok=True)
    logger = Logger()
    logger.save_agent_info(folder_path, agent, config, elapsed_time)
    if save:
        logger.save_trained_agent(agent, folder_path)
    logger.plot_scores(folder_path)
    print('\n--------------------------------')

def run_configuration(file, folder_path, agent_type, output_folder, total_timesteps, seed, save):
    env = TempRegulationEnv(start_from_random_day=True, seed=int(seed), max_steps_per_episode=4*24*14)
    logger = Logger()
    logger.reset()

    config = load_config(os.path.join(folder_path, file), agent_type, silent=True)
    agent = load_agent(env, agent_type, config)

    training_beginning = time.time()
    agent.train(total_timesteps)
    elapsed_time = time.time() - training_beginning
    print(f"Training time: {elapsed_time / 60} minutes.")

    last_avg_score = logger.get_last_avg_score()
    print(f">>>{file.removesuffix('.yaml')}+score:{last_avg_score:.2f}")
    save_agent_info(f"{output_folder}/{file.removesuffix('.yaml')}", agent, config, elapsed_time, save)

    # Copy config file to the output folder
    os.system(f"cp {os.path.join(folder_path, file)} {output_folder}/config.yaml")

    env.close()
    del agent, env

    return file, last_avg_score

def handle_interrupt(signum, frame, args, start_time, best_last_avg_score):
    print('\nInterrupted.')
    total_time = time.time() - start_time
    print_summary(best_last_avg_score, total_time, args.output, args.skip + len(best_last_avg_score), args.rpc)
    print('Exiting gracefully after interruption.')
    exit(1)

def run_configurations(args):
    signal.signal(signal.SIGINT, lambda signum, frame: handle_interrupt(signum, frame, args, start_time, best_last_avg_score))

    files = [f for f in os.listdir(args.folder) if f.endswith('.yaml')]
    if not files:
        print(f"Error: No files found in '{args.folder}'.")
        return

    start_time = time.time()
    best_last_avg_score = {}

    for index, file in enumerate(files[args.skip:], start=args.skip+1):
        print(f"Configuration {index} of {len(files)}: {file}")
        _, score = run_configuration(file, args.folder, args.agent, args.output, args.timesteps, args.seed, args.s)
        best_last_avg_score[file] = score

    total_time = time.time() - start_time
    print_summary(best_last_avg_score, total_time, args.output, len(files), args.rpc)

def print_summary(best_last_avg_score, total_time, output_folder, total_files, rpc=50):
    sorted_scores = sorted(best_last_avg_score.items(), key=lambda x: x[1], reverse=True)
    summary_path = f"{output_folder}/summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Summary:\nTotal time: {int(total_time//3600)} hours, {int(total_time%3600//60)} minutes, {total_time%60:.2f} seconds\n")
        f.write(f"Total configurations run {len(sorted_scores)} of {total_files}\n\n")
        f.write("Configurations results from best to worst:\n")
        for i, (config, score) in enumerate(sorted_scores, start=1):
            f.write(f"{i}. Configuration: {config.removesuffix('.yaml')}+score:{score:.2f}\n")

    print('Summary generated:', summary_path)
    generate_latex_table(summary_path, output_folder)

def generate_latex_table(summary_path, output_folder):
    latex_table = glt.generate_latex_table(summary_path)
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    latex_table_path = f"{output_folder}/table_{current_time}.tex"
    with open(latex_table_path, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table generated at {latex_table_path}")

def main():
    parser = argparse.ArgumentParser(description='Run configurations for DQL and PPO agents.')
    parser.add_argument('--agent', choices=['DQL', 'PPO'], required=True, help='Agent type to run configurations for')
    parser.add_argument('--folder', default='generated_configs', help='Folder with configurations')
    parser.add_argument('--timesteps', type=int, default=4*24*360*20, help='Total timesteps to train the agent')
    parser.add_argument('--output', default='generated_configs/results', help='Output folder to save results')
    parser.add_argument('--skip', type=int, default=0, help='Skip first n configurations')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the environment')
    parser.add_argument('--rpc', required=False, default=50, type=int, help='Rows per column in the table')
    parser.add_argument('-s', action='store_true', help='Save learned models for all configurations')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run_configurations(args)

if __name__ == '__main__':
    main()
