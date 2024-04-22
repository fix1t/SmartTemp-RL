import yaml
import torch

DQL_DEFAULT_CONFIG = {
    'network': {
        'seed': 42,
        'hidden_layers': [128, 32, 128],
        'activation': 'ReLU',
        'output_activation': 'Softmax'
    },
    'hyperparameters': {
        'learning_rate': 0.0005,
        'discount_factor': 0.99,
        'batch_size': 100,
        'replay_buffer_size': 100000,
        'interpolation_parameter': 0.001,
        'target_update_frequency': 4
    }
}

PPO_DEFAULT_CONFIG = {
    'network': {
        'seed': 42,
        'hidden_layers': [32, 128, 256],
        'activation': 'ReLU',
        'output_activation': 'Softmax'
    },
    'hyperparameters': {
        'batch_size': 2048,
        'max_timesteps_per_episode': 200,
        'discount_factor': 0.99,
        'n_updates_per_iteration': 10,
        'learning_rate': 3e-4,
        'clip': 0.2
        }
}

def log_progress(message, silent=False):
    """
    Log a progress message to the console.

    Parameters:
        message (str): The message to log.
        silent (bool): Whether to suppress the message.
    """
    if not silent:
        print(f"Config loader: {message}")

def load_config(file_path, algorithm='DQL', silent=False):
    """
    Load configuration from a YAML file, validate completeness, and set defaults for missing values.

    Parameters:
        file_path (str): Path to the YAML configuration file.
        algorithm (str): The algorithm to load the config for ('DQL' or 'PPO').

    Returns:
        dict: Configuration dictionary with all necessary values set, using defaults where required.
    """
    log_progress('---------------CONFIG---------------', silent=silent)
    config = DQL_DEFAULT_CONFIG.copy() if algorithm == 'DQL' else PPO_DEFAULT_CONFIG.copy()

    if file_path != '':
        load = True

        try:
            with open(file_path, 'r') as file:
                loaded_config = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            log_progress(f"Warning: Config file not found at '{file_path}'. Using default values.", silent=silent)
            load = False

        log_progress(f"Loading config for {algorithm} from '{file_path}'", silent=silent)

        if not isinstance(loaded_config, dict):
            log_progress("Warning: Loaded config is empty or invalid. Using default values.", silent=silent)
            load = False

        # Update the config with values from the loaded config, if they exist
        if load:
            for section in config:
                if section in loaded_config:
                    for key in config[section]:
                        if key in loaded_config[section]:
                            config[section][key] = loaded_config[section][key]
                        else:
                            log_progress(f"Warning: '{key}' not found in '{section}' section. Using default value.", silent=silent)
                else:
                    log_progress(f"Warning: '{section}' section is missing. Using default values.", silent=silent)

    else: # No config file provided
        log_progress("Warning: No config file provided. Using default values.", silent=silent)

    # Set the activation functions based on the names
    config['network']['activation'] = activation_function(config['network']['activation'])
    config['network']['output_activation'] = activation_function(config['network']['output_activation'])
    print(f"Configuration loaded: {config}")
    log_progress('------------------------------------', silent=silent)
    return config

def activation_function(name):
    """
    Get the activation function from the name.
    Parameters:
        name (str): Name of the activation function.
    Returns:
        A PyTorch activation function.
    """
    activations = {
        'ReLU': torch.nn.ReLU,
        'Sigmoid': torch.nn.Sigmoid,
        'Tanh': torch.nn.Tanh,
        'Softmax': lambda : torch.nn.Softmax(dim=-1)
    }
    if name in activations:
        return activations[name]
    else:
        print(f"Warning: Activation function '{name}' not recognized. Defaulting to ReLU.")
        return torch.nn.ReLU()


if __name__ == '__main__':
    config = load_config('algorithms/dql/configurations/default.yaml', 'DQL')
    print(config)
