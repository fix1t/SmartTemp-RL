"""
    File: config_generator.py
    Author: Gabriel Biel

    Description: Generate configurations for DQL, PPO agents. These configurations
    are used to train the agents with different hyperparameters and network configurations,
    to find the best performing agent configuration.
"""

import argparse
import yaml
import itertools
import os
nn_parameters = {
    "hidden_layers": [32, 64, 128, 256],
}

dql_parameters = {
    "learning_rate": [0.0005, 0.0015],
    "discount_factor": [0.90, 0.95, 0.99],
    "batch_size": [64, 96, 128],
    "learning_freqency": [4, 8, 16],
    "interpolation_parameter": [0.001, 0.005, 0.025],
}

ppo_parameters = {
    "learning_rate" : [0.0005, 0.0015],
    "discount_factor" : [0.90, 0.95, 0.99],
    "batch_size" : [1024, 2048, 4096],
    "n_updates_per_iteration" : [5, 10, 20],
    "clip" : [0.1, 0.2, 0.3],
}

dql_top_parameters = {
    "hidden_layers": [[32,256],[32,32,128],[256,128,256],[32,64,32],[32,64]],
    "hyperparametrs": [
        {
        "learning_rate": 0.0015,
        "batch_size": 96,
        "discount_factor": 0.95,
        "learning_freqency": 16,
        "interpolation_parameter": 0.005,
    },
        {
        "learning_rate": 0.0005,
        "batch_size": 128,
        "discount_factor": 0.95,
        "learning_freqency": 16,
        "interpolation_parameter": 0.001,
    },
        {
        "learning_rate": 0.0005,
        "batch_size": 96,
        "discount_factor": 0.95,
        "learning_freqency": 4,
        "interpolation_parameter": 0.001,
    },
        {
        "learning_rate": 0.0005,
        "batch_size": 96,
        "discount_factor": 0.95,
        "learning_freqency": 4,
        "interpolation_parameter": 0.025,
    },
        {
        "learning_rate": 0.0005,
        "batch_size": 128,
        "discount_factor": 0.99,
        "learning_freqency": 4,
        "interpolation_parameter": 0.025,
    }
    ]
}

ppo_top_parameters = {
    "hidden_layers": [[32,128,256],[128,32,256],[256,128,128],[128,256,64],[128,256,128]],
    "hyperparametrs": [
        {
        "learning_rate": 0.0005,
        "discount_factor": 0.95,
        "batch_size": 4096,
        "n_updates_per_iteration": 20,
        "clip": 0.3,
    },
        {
        "learning_rate": 0.0015,
        "discount_factor": 0.95,
        "batch_size": 4096,
        "n_updates_per_iteration": 10,
        "clip": 0.2,
    },
        {
        "learning_rate": 0.0005,
        "discount_factor": 0.95,
        "batch_size": 1024,
        "n_updates_per_iteration": 10,
        "clip": 0.3,
    },
        {
        "learning_rate": 0.0015,
        "discount_factor": 0.95,
        "batch_size": 1024,
        "n_updates_per_iteration": 10,
        "clip": 0.3,
    },
        {
        "learning_rate": 0.0005,
        "discount_factor": 0.90,
        "batch_size": 4096,
        "n_updates_per_iteration": 20,
        "clip": 0.3,
    }
    ]
}

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
        'learning_freqency': 'hyperparameters',
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

def yield_top_dynamic_configuration(dql_top_parameters):
    """Yield configurations combining hidden layers with specific hyperparameters from dql_top_parameters."""

    # Iterate through each hidden layer configuration
    for layer_config in dql_top_parameters["hidden_layers"]:

        # Iterate through each hyperparameters set corresponding to hidden layers
        for hyperparam_config in dql_top_parameters["hyperparametrs"]:
            config = {
                'network': {
                    'hidden_layers': layer_config
                },
                'hyperparameters': hyperparam_config
            }

            # Generate a filename or identifier for each configuration
            layer_details = f"layers:[{','.join(map(str, layer_config))}]"
            hyper_details = '+'.join([f"{key}:{value}" for key, value in hyperparam_config.items()])
            file_name = f"{layer_details}+{hyper_details}.yaml"

            # Yield the combined configuration and the filename
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
    hidden_layers = nn_parameters['hidden_layers']
    # generate combinations of hidden layers - 1 to 3 hidden layers with 32, 64, 128, 256 neurons
    for length in range(1, 4):  # Iterate over lengths from 1 to 3
        for hidden_layer in itertools.product(hidden_layers, repeat=length):
            config = base_config.copy()
            config['network']['hidden_layers'] = list(hidden_layer)
            layer_str = '[' + ','.join(map(str, hidden_layer)) + ']'
            file_name = 'hidden_layers:' +layer_str+ ".yaml"
            yield config, file_name

def generate_hp_configurations(agent_type, output_dir=None, top=False):
    if output_dir is None:
        output_dir = f"out/generated_configs/{agent_type.lower()}"

    agent_type = agent_type.upper()

    if top:
        parameters = dql_top_parameters if agent_type == "DQL" else ppo_top_parameters if agent_type == "PPO" else None
    else:
        parameters = dql_parameters if agent_type == "DQL" else ppo_parameters if agent_type == "PPO" else None

    if parameters is None:
        raise ValueError(f"Unknown agent type '{agent_type}'.")

    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for config, filen_name in yield_dynamic_configuration(parameters) if not top else yield_top_dynamic_configuration(parameters):
        index += 1
        print(f"Generated configuration {index} for {agent_type} to {filen_name}")
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
    parser = argparse.ArgumentParser(description="Generate configurations for DQL, PPO, and NN agents.")
    parser.add_argument('--agent', choices=['DQL', 'PPO', 'dql', 'ppo'], required=True, help='Agent type to run configurations for')
    parser.add_argument("--output", default='out/generated_configs', type=str, help="The output directory to save the generated configurations.")
    parser.add_argument("--top", action="store_true", help="Generate top configurations with predefined hyperparameters.")
    parser.add_argument("--hp", action="store_true", help="Generate hp configurations with predefined hyperparameters.")
    parser.add_argument("--nn", action="store_true", help="Generate nn configurations with predefined hyperparameters.")
    args = parser.parse_args()

    if args.hp:
        generate_hp_configurations(args.agent, args.output, top=False)
    elif args.top:
        generate_hp_configurations(args.agent, args.output, top=True)
    elif args.nn:
        generate_nn_configurations(args.output)
    else:
        raise ValueError("Please specify a configuration type to generate: --hp, --top, or --nn.")
