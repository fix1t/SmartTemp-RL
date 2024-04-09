import argparse
import os
import time
from algorithms.tools.logger import Logger
from tools.config_generator import generate_configurations
from tools.config_loader import load_config
from env.environment import TempRegulationEnv
from main import load_agent

def save_agent_info(folder_path, agent, config, elapsed_time):
    os.makedirs(folder_path, exist_ok=True)
    Logger().save_agent_info(f"{folder_path}", agent, config, elapsed_time)
    Logger().plot_scores(f"{folder_path}")
    print('\n--------------------------------')

def run_configurations(folder_path, total_timesteps=4*24*360*15, agent_type='DQL', output_folder='generated_configs/results'):
    """
    Run all configurations in a folder.
    """
    if agent_type not in ['DQL', 'PPO']:
        print(f"Error: Unknown agent type '{agent_type}'.")
        return
    # Get all files in the folder
    files = os.listdir(folder_path)
    files = sorted(files)

    # dictionary to store the best last average score for each configuration
    best_last_avg_score = {}
    file = ''

    start_time = time.time()

    # Run each configuration
    try:
        for file in files:
            if file.endswith('.yaml'):
                print(f"Running configuration: {file}")

                env = TempRegulationEnv(
                    start_from_random_day=True,
                    seed=int(42),
                    max_steps_per_episode=4*24*14,
                )
                Logger().reset()

                config = load_config(os.path.join(folder_path, file), agent_type, silent=True)
                agent = load_agent(env, agent_type, config)

                # Train the agent
                training_beginning = time.time()
                agent.train(total_timesteps)
                elapsed_time = time.time() - training_beginning

                best_last_avg_score[file] = Logger().get_last_avg_score()

                save_agent_info(f"{output_folder}/{agent_type.lower()}/{file.removesuffix('.yaml')}", agent, config, elapsed_time)

    except KeyboardInterrupt:
        print('\nInterrupted.')
        elapsed_time = time.time() - training_beginning
        best_last_avg_score[file] = Logger().get_last_avg_score()

        save_agent_info(f"{output_folder}/{agent_type.lower()}/{file.removesuffix('.yaml')}", agent, config, elapsed_time)

    finally:
        total_time = time.time() - start_time
        sorted_scores = sorted(best_last_avg_score.items(), key=lambda x: x[1], reverse=True)

        with open(f"{output_folder}/{agent_type.lower()}/summary.txt", 'w') as f:
            f.write('Summary:\n')
            f.write(f"Total time: {int(total_time//3600)} hours, {int(total_time%3600//60)} minutes, {total_time%60:.2f} seconds\n")
            f.write(f"Total configurations run {len(sorted_scores)} of {len(files)}\n\n")

            f.write("Configurations results from best to worst:\n")
            for i, (config, score) in enumerate(sorted_scores, start=1):
                f.write(f"{i}. Configuration: {config}, Average score of last 10: {score:.2f}\n")

        print('Summary:')
        print(f"Total time: {int(total_time//3600)} hours, {int(total_time%3600//60)} minutes, {int(total_time%60)} seconds")
        print("Top 5 configurations with the best last average scores:")
        for i, (config, score) in enumerate(sorted_scores[:5], start=1):
            print(f"{i}. Configuration: {config}, Average score of last 10: {score:.2f}")

        print('--------------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run configurations for DQL and PPO agents.')
    parser.add_argument('--agent', choices=['DQL', 'PPO'], required=True, type=str, help='Agent type to run configurations for')
    parser.add_argument('--folder', required=True, type=str, help='Folder with configurations')
    parser.add_argument('--timesteps', required=False, default=4*24*360*20, type=int, help='Total timesteps to train the agent')
    parser.add_argument('--output', required=False, default='generated_configs/results', type=str, help='Output folder to save results')
    args = parser.parse_args()

    generate_configurations(args.agent, args.folder)
    run_configurations(args.folder, args.timesteps, agent_type=args.agent, output_folder=args.output)
