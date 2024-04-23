import argparse
import os
import time
from algorithms.tools.logger import Logger
from tools.config_generator import generate_nn_configurations
from tools.config_generator import generate_hp_configurations
from tools.config_loader import load_config
from env.environment import TempRegulationEnv
from main import load_agent
import tools.generate_latex_table as glt

def save_agent_info(folder_path, agent, config, elapsed_time):
    os.makedirs(folder_path, exist_ok=True)
    Logger().save_agent_info(f"{folder_path}", agent, config, elapsed_time)
    Logger().plot_scores(f"{folder_path}")
    print('\n--------------------------------')

def run_configurations(folder_path, total_timesteps=4*24*360*15, agent_type='DQL', output_folder='generated_configs/results', skip=0):
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
        total_files = len(files)
        current_file = 0
        for file in files:

            if file.endswith('.yaml'):
                current_file += 1


                if skip > 1:
                    print(f"Configuration {current_file} of {total_files}: {file} -- Skipped")
                    skip -= 1
                    continue
                else:
                    print(f"Configuration {current_file} of {total_files}: {file}")

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
                print(f"Training time: {elapsed_time/60} minutes.")

                best_last_avg_score[file] = Logger().get_last_avg_score()
                save_agent_info(f"{output_folder}/{agent_type.lower()}/{file.removesuffix('.yaml')}", agent, config, elapsed_time)

    except KeyboardInterrupt:
        print('\nInterrupted.')
        elapsed_time = time.time() - training_beginning
        print(f"Training time: {elapsed_time/60} minutes.")

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

def generate_table(summary_path, output_folder, rpc, nn=False, isDql=True):

    latex_table = glt.generate_latex_table(summary_path, nn, isDql, rpc)

    current_time = time.strftime('%Y-%m-%d_%H-%M-%S')

    with open(f"{output_folder}/table_{current_time}.tex", "w") as f:
        f.write(latex_table)

    print(f"LaTeX table generated at {output_folder}/table_{current_time}.tex")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run configurations for DQL and PPO agents.')
    parser.add_argument('--agent', choices=['DQL', 'PPO'], required=True, type=str, help='Agent type to run configurations for')
    parser.add_argument('--folder', required=True, type=str, help='Folder with configurations')
    parser.add_argument('--timesteps', required=False, default=4*24*360*20, type=int, help='Total timesteps to train the agent')
    parser.add_argument('--output', required=False, default='generated_configs/results', type=str, help='Output folder to save results')
    parser.add_argument('--skip', required=False, default=0, type=int, help='Skip fir n configurations')
    parser.add_argument('-nn', action='store_true', help='Flag to parse neural network configurations')
    args = parser.parse_args()

    if args.nn:
        generate_nn_configurations(args.folder)
    else:
        generate_hp_configurations(args.agent, args.folder)

    os.makedirs(args.output, exist_ok=True)
    run_configurations(args.folder, args.timesteps, agent_type=args.agent, output_folder=args.output, skip=args.skip)

    # Generate table
    rows_per_column = 50
    table_output_folder = 'out/generated_configs/tables'
    os.makedirs(table_output_folder, exist_ok=True)

    generate_table(f"{args.output}/{args.agent.lower()}/summary.txt", table_output_folder, rows_per_column, args.nn, args.agent == 'DQL')
