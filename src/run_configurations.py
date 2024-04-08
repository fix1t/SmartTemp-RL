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
        top_5 = sorted(best_last_avg_score.items(), key=lambda x: x[1], reverse=True)[:5]

        with open(f"{output_folder}/{agent_type.lower()}/summary.txt", 'w') as f:
            f.write('Summary:\n')
            f.write(f"Total time: {int(total_time//3600)} hours, {int(total_time%3600//60)} minutes, {total_time%60:.2f} seconds\n")

            top_5 = sorted(best_last_avg_score.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("Top 5 configurations with the best last average scores:\n")
            for i, (config, score) in enumerate(top_5, start=1):
                f.write(f"{i}. Configuration: {config}, Average score of last 10: {score:.2f}\n")

        print('Summary:')
        print(f"Total time: {int(total_time//3600)} hours, {int(total_time%3600//60)} minutes, {int(total_time%60)} seconds")
        print("Top 5 configurations with the best last average scores:")
        for i, (config, score) in enumerate(top_5, start=1):
            print(f"{i}. Configuration: {config}, Average score of last 10: {score:.2f}")

        print('--------------------------------')

if __name__ == '__main__':
    generate_configurations('DQL')
    generate_configurations('PPO')
    run_configurations('generated_configs/dql', total_timesteps=4*24*365*20, agent_type='DQL')
    run_configurations('generated_configs/ppo', total_timesteps=4*24*365*20, agent_type='PPO')
