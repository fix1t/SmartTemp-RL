"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
Strongly inspired by MorvanZhou's code: https://github.com/MorvanZhou/pytorch-A3C/tree/master
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
from smart_home_env import SmartHomeTempControlEnv

# Limit the number of threads to avoid potential overhead in multiprocessing environments
os.environ["OMP_NUM_THREADS"] = "1"

# Hyperparameters
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

# Environment setup
env = SmartHomeTempControlEnv(start_from_random_day=False)

N_S = env.observation_space.shape[0]
N_A = env.action_space.n
print(f"Observation space: {N_S}, Action space: {N_A}")

def debug_log(message):
    """
    Debug function to log messages for troubleshooting.

    Args:
        message (str): The debug message to be logged.
    """
    return
    print(f"DEBUG: {message}")

class Net(nn.Module):
    """
    Neural Network architecture for both policy (actor) and value (critic) estimation.
    """
    def __init__(self, s_dim, a_dim):
        """
        Initialize the neural network with separate pathways for policy and value.

        Args:
            s_dim (int): Dimension of the state space.
            a_dim (int): Dimension of the action space.
        """
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        self.initialize_weights([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): The input state.

        Returns:
            Tuple[Tensor, Tensor]: The logits for the policy and the value estimate.
        """
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        """
        Choose an action based on the policy logits.

        Args:
            s (Tensor): The current state.

        Returns:
            int: The selected action.
        """
        self.eval()  # Set the network to evaluation mode
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        """
        Calculate the loss for the network.

        Args:
            s (Tensor): States.
            a (Tensor): Actions taken.
            v_t (Tensor): Target values.

        Returns:
            Tensor: The calculated loss.
        """
        self.train()  # Set the network to training mode
        logits, values = self.forward(s)
        td = v_t - values  # Temporal difference error
        c_loss = td.pow(2)  # Critic loss (value loss)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()  # Detach td to stop gradients
        a_loss = -exp_v  # Actor loss
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    @staticmethod
    def initialize_weights(layers):
        """
        Initialize the weights of the network layers.

        Args:
            layers (list): List of network layers to initialize.
        """
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

class Worker(mp.Process):
    """
    Worker process for A3C training.
    """
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        """
        Initialize a worker process.

        Args:
            gnet (Net): The global network.
            opt (torch.optim.Optimizer): The optimizer.
            global_ep (mp.Value): Counter for global episodes.
            global_ep_r (mp.Value): Global episode reward.
            res_queue (mp.Queue): Queue to put results for logging.
            name (int): The name/ID of the worker.
        """
        super(Worker, self).__init__()
        self.name = f'w{int(name):02d}'
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = SmartHomeTempControlEnv(start_from_random_day=False).unwrapped
        # if self.name == 'w00':
        #             self.env.render()

    def run(self):
        """
        Run the worker process, executing episodes and updating the global network.
        """
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            debug_log(f"inital observation: {s}")
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                debug_log(f"action: {a}")
                s_, r, done, _ = self.env.step(a)
                debug_log(f"step: {s_}, {r}, {done}, _")
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())] # mp.cpu_count()
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()


