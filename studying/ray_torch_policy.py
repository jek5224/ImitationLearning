from typing import Dict

from ray.rllib.algorithms.ppo.ppo_torch_policy import *
from ray.rllib.evaluation.postprocessing import *

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

def custom_compute_gae_for_sample_batch(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[Episode] = None,
) -> SampleBatch:
    
    if (
        sample_batch[SampleBatch.DONES][-1]
        and sample_batch[SampleBatch.INFOS][-1]["end"] != 3
    ):
        last_r = 0.0
    else:
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        last_r = policy._value(**input_dict)

    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
    )

    return batch


class CustomPPOTorchPolicy(PPOTorchPolicy):
    def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            return custom_compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )