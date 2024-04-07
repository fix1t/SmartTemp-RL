import yaml
import torch

DQL_DEFAULT_CONFIG = {
    'network': {
        'seed': 42,
        'hidden_layers': [64, 64],
        'activation': 'ReLU',
        'output_activation': 'Softmax'
    },
    'hyperparameters': {
        'learning_rate': 0.0005,
        'discount_factor': 0.99,
        'epsilon': 0.1,
        'batch_size': 32,
        'epsilon_starting_value': 1.0,
        'epsilon_ending_value': 0.01,
        'epsilon_decay_value': 0.995,
        'minibatch_size': 100,
        'replay_buffer_size': 100000,
        'interpolation_parameter': 0.001
    }
}

PPO_DEFAULT_CONFIG = {
    'network': {
        'seed': 42,
        'hidden_layers': [64, 64],
        'activation': 'ReLU',
        'output_activation': 'Softmax'
    },
    'hyperparameters': {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': False,
        'render_every_i': 10
    }
}

def load_config(file_path, algorithm='DQL'):
    """
    Load configuration from a YAML file, validate completeness, and set defaults for missing values.

    Parameters:
        file_path (str): Path to the YAML configuration file.
        algorithm (str): The algorithm to load the config for ('DQL' or 'PPO').

    Returns:
        dict: Configuration dictionary with all necessary values set, using defaults where required.
    """
    print('--------------------------------')
    print(f"Loading config for {algorithm} from '{file_path}'")

    config = DQL_DEFAULT_CONFIG.copy() if algorithm == 'DQL' else PPO_DEFAULT_CONFIG.copy()

    try:
        with open(file_path, 'r') as file:
            loaded_config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"Warning: Config file not found at '{file_path}'. Using default values.")
        print('--------------------------------')
        return config

    if not isinstance(loaded_config, dict):
        print("Warning: Loaded config is empty or invalid. Using default values.")
        print('--------------------------------')
        return config

    # Update the config with values from the loaded config, if they exist
    for section in config:
        if section in loaded_config:
            for key in config[section]:
                if key in loaded_config[section]:
                    config[section][key] = loaded_config[section][key]
                else:
                    print(f"Warning: '{key}' not found in '{section}' section. Using default value.")
        else:
            print(f"Warning: '{section}' section is missing. Using default values.")

    # Set the activation functions based on the names
    config['network']['activation'] = activation_function(config['network']['activation'])
    config['network']['output_activation'] = activation_function(config['network']['output_activation'])

    print('--------------------------------')
    return config

def activation_function(name):
    """
    Get the activation function from the name.

    Parameters:
        name (str): Name of the activation function.

    Returns:
        torch.nn.Module: The activation function module.
    """
    if name == 'ReLU':
        return torch.nn.ReLU()
    elif name == 'Sigmoid':
        return torch.nn.Sigmoid()
    elif name == 'Tanh':
        return torch.nn.Tanh()
    elif name == 'Softmax':
        return torch.nn.Softmax(dim=1)
    else:
        print(f"Warning: Activation function '{name}' not found. Using ReLU.")
        return torch.nn.ReLU()

if __name__ == '__main__':
    config = load_config('/home/fix/SmartTemp-RL/src/configurations/ppo/default.yaml', 'PPO')
    print(config)
    config = load_config('/home/fix/SmartTemp-RL/src/configurations/dql/default.yaml', 'DQL')
    print(config)
    config = load_config('configurations/ppo/default.yaml', 'PPO')
    print(config)
    config = load_config('config.yml', 'invalid')
    print(config)
