import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train or test PPO/DQL model.')
    parser.add_argument('-m', '--mode', choices=['train', 'test'], required=False, default='train' , type=str ,help='Mode to run the script in. Test mode requires a model file.')
    parser.add_argument('-a', '--algorithm', choices=['PPO', 'DQL'], required=False, default='DQL', type=str ,help='Algorithm to use')
    parser.add_argument('--actor_model', required=False, default='', type=str ,help='Path to the actor model file - only for PPO')
    parser.add_argument('--critic_model', required=False, default='', type=str ,help='Path to the critic model file - only for PPO')
    parser.add_argument('--local_qnetwork', required=False, default='', type=str ,help='Path to the local qnetwork model file - only for DQL')
    parser.add_argument('--target_qnetwork', required=False, default='', type=str ,help='Path to the target qnetwork model file - only for DQL')
    parser.add_argument('--config', required=False, default='', type=str ,help='Path to the config file. Specifies hyperparameters and network parameters for algorithm')
    parser.add_argument('--seed', required=False, default='',  type=int, help='Seed for the environment')
    args = parser.parse_args()

    if args.seed == '':
        args.seed = None

    return args
