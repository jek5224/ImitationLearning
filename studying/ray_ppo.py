from typing import Type
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.rllib.policy.policy import Policy

from ray.rllib.algorithms.ppo import PPO as PPOTrainer

class CustomPPOTrainer(PPOTrainer):
    @override(Algorithm)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from ray_torch_policy import CustomPPOTorchPolicy
            return CustomPPOTorchPolicy