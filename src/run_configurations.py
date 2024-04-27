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

def run_configurations(folder_path, total_timesteps=4*24*360*15, agent_type='DQL',
                       output_folder='generated_configs/results', skip=0, seed=42):
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
                # Skip first n configurations
                if skip > 0:
                    print(f"Configuration {current_file} of {total_files}: {file} -- Skipped")
                    skip -= 1
                    continue
                else:
                    print(f"Configuration {current_file} of {total_files}: {file}")

                env = TempRegulationEnv(
                    start_from_random_day=True,
                    seed=int(seed),
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
                save_agent_info(f"{output_folder}/{file.removesuffix('.yaml')}", agent, config, elapsed_time)

    except KeyboardInterrupt:
        print('\nInterrupted.')
        elapsed_time = time.time() - training_beginning
        print(f"Training time: {elapsed_time/60} minutes.")

        best_last_avg_score[file] = Logger().get_last_avg_score()
        save_agent_info(f"{output_folder}/{file.removesuffix('.yaml')}", agent, config, elapsed_time)

    finally:
        total_time = time.time() - start_time
        sorted_scores = sorted(best_last_avg_score.items(), key=lambda x: x[1], reverse=True)

        with open(f"{output_folder}/summary.txt", 'w') as f:
            f.write('Summary:\n')
            f.write(f"Total time: {int(total_time//3600)} hours, {int(total_time%3600//60)} minutes, {total_time%60:.2f} seconds\n")
            f.write(f"Total configurations run {len(sorted_scores)} of {len(files)}\n\n")

            f.write("Configurations results from best to worst:\n")
            for i, (config, score) in enumerate(sorted_scores, start=1):
                #strip .yaml from the file name
                config = config.removesuffix('.yaml')

                f.write(f"{i}. Configuration: {config}+score:{score:.2f}\n")

        print('Summary:')
        print(f"Total time: {int(total_time//3600)} hours, {int(total_time%3600//60)} minutes, {int(total_time%60)} seconds")
        print("Top 5 configurations with the best last average scores:")
        for i, (config, score) in enumerate(sorted_scores[:5], start=1):
            print(f"{i}. Configuration: {config}, Average score of last 10: {score:.2f}")

        print('--------------------------------')

def generate_table(summary_path, output_folder, rpc, nn=False):

    latex_table = glt.generate_latex_table(summary_path, nn, rpc)

    current_time = time.strftime('%Y-%m-%d_%H-%M')

    with open(f"{output_folder}/table_{current_time}.tex", "w") as f:
        f.write(latex_table)

    print(f"LaTeX table generated at {output_folder}/table_{current_time}.tex")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run configurations for DQL and PPO agents.')
    parser.add_argument('--agent', choices=['DQL', 'PPO'], required=True, type=str, help='Agent type to run configurations for')
    parser.add_argument('--folder', required=False, type=str, help='Folder with configurations - else generates configurations')
    parser.add_argument('--timesteps', required=False, default=4*24*360*20, type=int, help='Total timesteps to train the agent')
    parser.add_argument('--output', required=False, default='generated_configs/results', type=str, help='Output folder to save results')
    parser.add_argument('--skip', required=False, default=0, type=int, help='Skip first n configurations')
    parser.add_argument('--rpc', required=False, default=50, type=int, help='Rows per column in the table')
    parser.add_argument('-nn', action='store_true', help='Flag to generate neural network configurations to --folder')
    parser.add_argument('-hp', action='store_true', help='Flag to generate hyperparameter configurations to --folder')
    parser.add_argument('-top', action='store_true', help='Flag to generate and parse top hp and nn configurations to --folder')

    parser.add_argument('--seed', required=False, default=42, type=int, help='Seed for the environment')
    args = parser.parse_args()

    if args.nn and args.hp :
        print("Error: Cannot parse neural network and hyperparameter configurations at the same time.")
        exit(1)
    elif args.top:
        generate_hp_configurations(args.agent, args.folder, top=True)
    elif args.nn:
        generate_nn_configurations(args.folder)
    elif args.hp:
        generate_hp_configurations(args.agent, args.folder)

    os.makedirs(args.output, exist_ok=True)

    # Run configurations from folder
    run_configurations(args.folder, args.timesteps, agent_type=args.agent, output_folder=args.output,
                       skip=args.skip, seed=args.seed)

    table_output_folder = 'out/generated_configs/tables' if not args.output else f"{args.output}"
    os.makedirs(table_output_folder, exist_ok=True)

    generate_table(f"{args.output}/summary.txt", table_output_folder, args.rpc, args.nn)
