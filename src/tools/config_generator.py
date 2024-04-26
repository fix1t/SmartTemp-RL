import yaml
import itertools
import os

dql_parameters = {
    "learning_rate": [0.0005, 0.0015],
    "discount_factor": [0.90, 0.95, 0.99],
    "batch_size": [64, 96, 128],
    "target_update_frequency": [5, 10, 20],
    "interpolation_parameter": [0.001, 0.005, 0.025],
}

ppo_parameters = {
    "learning_rate" : [0.0005, 0.0015],
    "discount_factor" : [0.90, 0.95, 0.99],
    "batch_size" : [1024, 2048, 4096],
    "n_updates_per_iteration" : [5, 10, 20],
    "clip" : [0.1, 0.2, 0.3],
}

nn_parameters = {
    "hidden_layers": [32, 64, 128, 256],
}

def yield_nn_configuration():
    base_config = {
        'network': {
            'hidden_layers': [64, 64],
            'activation': 'ReLU',
            'output_activation': 'Softmax',
            'seed': 42,
        }
    }
    hidden_layers = nn_parameters['hidden_layers']
    # generate combinations of hidden layers - 1 to 3 hidden layers with 32, 64, 128, 256 neurons
    for length in range(1, 4):  # Iterate over lengths from 1 to 3
        for hidden_layer in itertools.product(hidden_layers, repeat=length):
            config = base_config.copy()
            config['network']['hidden_layers'] = list(hidden_layer)
            layer_str = '[' + ','.join(map(str, hidden_layer)) + ']'
            file_name = 'hidden_layers:' +layer_str+ ".yaml"
            yield config, file_name

def initialize_sections_from_structure(config_structure):
    """Initialize configuration base with sections from the config structure."""
    return {section: {} for section in set(config_structure.values())}

def yield_dynamic_configuration(parameters):
    config_structure = {
        'learning_rate': 'hyperparameters',
        'batch_size': 'hyperparameters',
        'interpolation_parameter': 'hyperparameters',
        'replay_buffer_size': 'hyperparameters',
        'discount_factor': 'hyperparameters',
        'max_timesteps_per_episode': 'hyperparameters',
        'n_updates_per_iteration': 'hyperparameters',
        'target_update_frequency': 'hyperparameters',
        'clip': 'hyperparameters',

        'hidden_layers': 'network',
        'activation': 'network',
        'output_activation': 'network',
        'seed': 'network',
    }

    # Initialize base configuration dictionary
    config_base = initialize_sections_from_structure(config_structure)

    # Generate all combinations of parameter values
    param_keys = list(parameters.keys())
    param_values = [parameters[key] for key in param_keys]

    for combination in itertools.product(*param_values):
        config = {section: {k: v for k, v in zip(param_keys, combination) if config_structure[k] == section} for section in config_base}
        file_details = [f"{key}:{value}" for key, value in zip(param_keys, combination)]

        # Generate a filename based on the parameter values
        file_name =  '+'.join(file_details) + ".yaml"
        yield config, file_name

def generate_hp_configurations(agent_type, output_dir=None):
    if output_dir is None:
        output_dir = f"out/generated_configs/{agent_type.lower()}"

    if agent_type == "DQL":
        parameters = dql_parameters
    elif agent_type == "PPO":
        parameters = ppo_parameters
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'.")

    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for config, filen_name in yield_dynamic_configuration(parameters=parameters):
        index += 1
        if output_dir:
            with open(os.path.join(output_dir, filen_name), 'w') as file:
                yaml.dump(config, file)

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
