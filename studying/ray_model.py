import torch
import torch.nn as nn
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  # If Linear is inside classname
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

class SimulationNN(nn.Module):
    def __init__(self, num_states, num_actions, learningStd=False):
        nn.Module.__init__(self)
        self.num_states = num_states
        self.num_actions = num_actions

        self.num_h1 = 512
        self.num_h2 = 512
        self.num_h3 = 512

        self.log_std = None
        init_log_std = torch.ones(num_actions)

        if learningStd:
            self.log_std = nn.Parameter(init_log_std)
        else:
            self.log_std = init_log_std

        self.policy_fc = nn.Sequential(
            nn.Linear(self.num_states, self.num_h1),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h1, self.num_h2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h2, self.num_h3),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h3, self.num_actions),
        )

        self.value_fc = nn.Sequential(
            nn.Linear(self.num_states, self.num_h1),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h1, self.num_h2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h2, self.num_h3),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_h3, 1),
        )

        self.reset()

        if torch.cuda.is_available():
            if not learningStd:
                self.log_std = self.log_std.cuda()
            self.cuda()

    def reset(self):
        self.policy_fc.apply(weights_init)
        self.value_fc.apply(weights_init)

    def forward(self, x):
        p_out = MultiVariateNormal(self.policy_fc.forward(x), self.log_std.exp())
        v_out = self.value_fc.forward(x)
        return p_out, v_out
    
    def load(self, path):
        print("load simulation nn {}".format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    def save(self, path):
        print("save simulation nn {}".format(path))
        torch.save(self.state_dict(), path)

    def get_action(self, s):
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.loc.cpu().detach().numpy()
    
    def get_value(self, s):
        ts = torch.tensor(s)
        _, v = self.forward(ts)
        return v.cpu().detach().numpy()
    
    def get_random_action(self, s):
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.sample().cpu().detach().numpy()
    
    def get_noise(self):
        return self.log_std.exp().mean().item()

class SimulationNN_Ray(TorchModelV2, SimulationNN):
    def __init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        num_states = np.prod(obs_space.shape)
        num_actions = np.prod(action_space.shape)
        SimulationNN.__init__(self, num_states, num_actions)
        TorchModelV2.__init__(  # model_config, name
            self, obs_space, action_space, num_outputs, {}, "SimulationNN_Ray"
        )
        num_outputs = 2 * num_actions
        self._value = None

    def get_value(self, obs):
        with torch.no_grad():
            _, v  = SimulationNN.forward(self, obs)
            return v
        
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)
        action_dist, self._value = SimulationNN.forward(self, x)
        action_tensor = torch.cat([action_dist.loc, action_dist.scale.log()], dim=1)
        return action_tensor, state
    
    def value_function(self):
        return self._value.squeeze(1)
    
    def reset(self):
        SimulationNN.reset(self)

    def vf_reset(self):
        SimulationNN.vf_reset(self)

    def pi_reset(self):
        SimulationNN.pi_reset(self)


class PolicyNN:
    def __init__(
            self,
            num_states,
            num_actions,
            policy_state,
            filter_state,
            device,
            learningStd=False,
    ):
        self.policy = SimulationNN(num_states, num_actions, learningStd).to(device)

        self.policy.log_std = self.policy.log_std.to(device)
        self.policy.load_state_dict(convert_to_torch_tensor(policy_state))
        self.policy.eval()
        self.filter = filter_state

    def get_filter(self):
        return self.filter.copy()
    
    def get_value(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtye=np.float32)
        v = self.policy.get_value(obs)
        return v
    
    def get_value_function_weight(self):
        return self.policy.value_fc.state_dict()
    
    def get_action(self, obs, is_random=False):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return (
            self.policy.get_action(obs)
            if not is_random
            else self.policy.get_random_action(obs)
        )
    
    def get_filtered_obs(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return obs
    
    def weight_filter(self, unnormalized, beta):
        scale_factor = 1000.0
        return torch.sigmoid(
            torch.tensor([scale_factor * (unnormalized - beta)])
        ).numpy()[0]
    
    def state_dict(self):
        state = {}
        state["weight"] = self.policy.state_dict()
        state["filter"] = self.filter
        return state
    
    # def soft_load_state_dict(self, _state_dict):
    #     self.policy.soft_load_state_dict(_state_dict)

import pickle
def loading_network(
        path,
        num_states=0,
        num_actions=0,
        device="cpu",
):
    state = pickle.load(open(path, "rb"))
    worker_state = pickle.loads(state["worker"])
    policy_state = worker_state["state"]["default_policy"]["weights"]
    filter_state = worker_state["filters"]["default_policy"]
    device = torch.device(device)
    learningStd = "log_std" in policy_state.keys()
    policy = PolicyNN(
        num_states, num_actions, policy_state, filter_state, device, learningStd
    )

    return policy