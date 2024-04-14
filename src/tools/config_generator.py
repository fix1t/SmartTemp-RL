import yaml
import itertools
import os

def generate_configurations(agent_type, output_dir=None):
    # Define base configurations for DQL and PPO
    base_configs = {
        'DQL': {
            'network': {
                'seed': 42,
                'hidden_layers': [64, 64],
                'activation': "ReLU",
                'output_activation': "Softmax"
            },
            'hyperparameters': {
                'learning_rate': 0.0005,
                'discount_factor': 0.99,
                'batch_size': 100,
                'replay_buffer_size': 100000,
                'interpolation_parameter': 0.001,
                'learn_every_n_steps': 4
            }
        },
        'PPO': {
            'network': {
                'seed': 42,
                'hidden_layers': [64, 64],
                'activation': "ReLU",
                'output_activation': "Softmax"
            },
            'hyperparameters': {
                'learning_rate': 0.0003,
                'batch_size': 2048,
                'max_timesteps_per_episode': 200,
                'discount_factor': 0.99,
                'n_updates_per_iteration': 10,
                'clip': 0.2
            }
        }
    }

    base_config = base_configs[agent_type]

    learning_rates = [0.001, 0.0015, 0.002, 0.0025]
    discount_factors = [0.90, 0.95, 0.99, 0.995]

    # DQL specific hyperparameters
    batch_sizes_dql = [64, 128, 256, 512]
    learn_every_n_steps = [2, 4, 8]

    # PPO specific hyperparameters
    n_updates_per_iteration = [5, 10, 20]
    batch_sizes_ppo = [512, 1024, 2048, 4096]

    if agent_type == 'DQL':
        combinations = list(itertools.product(learning_rates, batch_sizes_dql, discount_factors, learn_every_n_steps))
    else:  # For PPO
        combinations = list(itertools.product(learning_rates, batch_sizes_ppo, discount_factors, n_updates_per_iteration))

    # Generate combinations

    if output_dir is None:
        output_dir = f"generated_configs/{agent_type.lower()}"

    os.makedirs(output_dir, exist_ok=True)

    # Generate and save configurations
    for i, (lr, batch_size, discount_factor) in enumerate(combinations, start=1):
        config = base_config.copy()
        config['hyperparameters']['learning_rate'] = lr
        if agent_type == 'DQL':
            config['hyperparameters']['batch_size'] = batch_size
        else:  # For PPO
            config['hyperparameters']['batch_size'] = batch_size
        config['hyperparameters']['discount_factor'] = discount_factor

        config_filename = f"cfg_lr_{lr:.4f}_bs_{batch_size}_df_{discount_factor}.yaml"
        with open(os.path.join(output_dir, config_filename), 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Configuration {i:03d} for {agent_type} saved to {config_filename}")

    print(f"Generated {i} configurations for {agent_type} in '{output_dir}' directory.")
