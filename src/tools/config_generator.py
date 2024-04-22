import yaml
import itertools
import os

def yield_dql_hp_configuration():
    base_config = {
        'hyperparameters': {
            'learning_rate': 0.0005,
            'discount_factor': 0.99,
            'batch_size': 100,
            'replay_buffer_size': 100000,
            'interpolation_parameter': 0.001,
            'target_update_frequency': 4
        }
    }

    learning_rates = [0.0005, 0.0015]
    discount_factors = [0.90, 0.95, 0.99]
    batch_sizes = [128, 256, 512]
    target_update_frequency = [5, 10, 20]
    interpolation_parameters = [0.001, 0.005, 0.025]

    # Generate combinations
    for lr, batch_size, discount_factor, n, i in itertools.product(learning_rates, batch_sizes, discount_factors, target_update_frequency, interpolation_parameters):
        config = base_config.copy()
        config['hyperparameters']['learning_rate'] = lr
        config['hyperparameters']['batch_size'] = batch_size
        config['hyperparameters']['discount_factor'] = discount_factor
        config['hyperparameters']['target_update_frequency'] = n
        config['hyperparameters']['interpolation_parameter'] = i
        file_name = f"dql_lr_{lr:.4f}_bs_{batch_size}_df_{discount_factor:.2f}_nu_{n}_it_{i:.3f}.yaml"
        yield config, file_name

def yield_ppo_hp_configuration():
    base_config = {
        'hyperparameters': {
            'learning_rate': 0.0003,
            'batch_size': 2048,
            'max_timesteps_per_episode': 200,
            'discount_factor': 0.99,
            'n_updates_per_iteration': 10,
            'clip': 0.2
        }
    }

    learning_rates = [0.0005, 0.0015]
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
        file_name = f"ppo_lr_{lr:.4f}_bs_{batch_size}_df_{discount_factor:.2f}_nu_{n}_cl_{c:.1f}.yaml"
        yield config, file_name

def yield_nn_configuration():
    base_config = {
        'network': {
            'hidden_layers': [64, 64],
            'activation': 'ReLU',
            'output_activation': 'Softmax',
            'seed': 42,
        }
    }
    hidden_layers = [32, 64, 128, 256]

    # generate combinations of hidden layers - 1 to 3 hidden layers with 32, 64, 128, 256 neurons
    for length in range(1, 4):  # Iterate over lengths from 1 to 3
        for hidden_layer in itertools.product(hidden_layers, repeat=length):
            config = base_config.copy()
            config['network']['hidden_layers'] = list(hidden_layer)
            file_name = 'nn_' + '_'.join(map(str, hidden_layer)) + ".yaml"
            yield config, file_name

def generate_hp_configurations(agent_type, output_dir=None):
    if output_dir is None:
        output_dir = f"out/generated_configs/{agent_type.lower()}"

    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for config, filen_name in yield_dql_hp_configuration() if agent_type == "DQL" else yield_ppo_hp_configuration():
        index += 1
        if output_dir:
            with open(os.path.join(output_dir, filen_name), 'w') as file:
                yaml.dump(config, file)
        print(f"Generated configuration {index} for {agent_type} to {filen_name}")

def generate_nn_configurations(output_dir=None):
    if output_dir is None:
        output_dir = f"out/generated_configs/nn"

    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for config, filen_name in yield_nn_configuration():
        index += 1
        if output_dir:
            with open(os.path.join(output_dir, filen_name), 'w') as file:
                    yaml.dump(config, file)
        print(f"Generated configuration {index} for NN to {filen_name}")

if __name__ == "__main__":
    generate_hp_configurations("DQL")
    generate_hp_configurations("PPO")
    generate_nn_configurations()
