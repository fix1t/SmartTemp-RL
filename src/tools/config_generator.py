import yaml
import itertools
import os

def yield_dql_configurations():
    base_config = {
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
    }

    learning_rates = [0.0005, 0.001, 0.002]
    discount_factors = [0.90, 0.95, 0.99]
    batch_sizes = [128, 256, 512]
    learn_every_n_steps = [5, 10, 20]
    interpolation_parameters = [0.001, 0.002, 0.003]

    # Generate combinations
    for lr, batch_size, discount_factor, n, i in itertools.product(learning_rates, batch_sizes, discount_factors, learn_every_n_steps, interpolation_parameters):
        config = base_config.copy()
        config['hyperparameters']['learning_rate'] = lr
        config['hyperparameters']['batch_size'] = batch_size
        config['hyperparameters']['discount_factor'] = discount_factor
        config['hyperparameters']['learn_every_n_steps'] = n
        config['hyperparameters']['interpolation_parameter'] = i
        file_name = f"dql_lr_{lr}_bs_{batch_size}_df_{discount_factor}_n_{n}_i_{i}.yaml"
        yield config, file_name

def yield_ppo_configurations():
    base_config = {
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

    learning_rates = [0.0003, 0.0005, 0.0007]
    discount_factors = [0.90, 0.95, 0.99]
    batch_sizes = [1024, 2048, 4096]
    n_updates_per_iteration = [5, 10, 20]
    clip = [0.1, 0.2, 0.3]

    # Generate combinations
    for lr, batch_size, discount_factor, n, c in itertools.product(learning_rates, batch_sizes, discount_factors, n_updates_per_iteration, clip):
        config = base_config.copy()
        config['hyperparameters']['learning_rate'] = lr
        config['hyperparameters']['batch_size'] = batch_size
        config['hyperparameters']['discount_factor'] = discount_factor
        config['hyperparameters']['n_updates_per_iteration'] = n
        config['hyperparameters']['clip'] = c
        file_name = f"ppo_lr_{lr}_bs_{batch_size}_df_{discount_factor}_n_{n}_c_{c}.yaml"
        yield config, file_name

def generate_configurations(agent_type, output_dir=None):
    if output_dir is None:
        output_dir = f"generated_configs/{agent_type.lower()}"

    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for config, filen_name in yield_dql_configurations() if agent_type == "DQL" else yield_ppo_configurations():
        index += 1
        if output_dir:
            with open(os.path.join(output_dir, filen_name), 'w') as file:
                yaml.dump(config, file)
        print(f"Generated configuration {index} for {agent_type} to {filen_name}")

if __name__ == "__main__":
    generate_configurations("DQL")
    generate_configurations("PPO")
