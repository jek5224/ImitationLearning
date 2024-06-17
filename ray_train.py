import argparse
import os
from pathlib import Path
from ray_model import SimulationNN_Ray
from env import Env as MyEnv

import ray
from ray import tune
from ray_ppo import CustomPPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
import pickle

torch, nn = try_import_torch()

w_reg_act = 0.01
w_reg_wo_relu = 0.01
w_target = 1.0


def create_my_trainer(rl_algorithm: str):
    if rl_algorithm == "PPO":
        RLTrainer = CustomPPOTrainer
    else:
        raise RuntimeError(f"Invalid algorithm {rl_algorithm}!")

    class MyTrainer(RLTrainer):
        def setup(self, config):
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )      
            self.trainer_config = config.pop("trainer_config")
            RLTrainer.setup(self, config=config)

            self.max_reward = -float("inf")
            self.idx = 0

            
        def step(self):
            result = RLTrainer.step(self)
            current_reward = result["episode_reward_mean"]
            result["sampler_results"].pop("hist_stats")


            if self.max_reward < current_reward:
                self.max_reward = current_reward
                self.save_max_checkpoint(self._logdir)
            self.save_last_checkpoint(self._logdir)
            self.idx += 1

            return result

        def __getstate__(self):
            state = RLTrainer.__getstate__(self)
            return state

        def __setstate__(self, state):
            RLTrainer.__setstate__(self, state)

        def save_checkpoint(self, checkpoint_path):
            print(f"Saving checkpoint at path {checkpoint_path}")
            RLTrainer.save_checkpoint(self, checkpoint_path)
            return checkpoint_path

        def save_max_checkpoint(self, checkpoint_path) -> str:
            with open(Path(checkpoint_path) / "max_checkpoint", "wb") as f:
                pickle.dump(self.__getstate__(), f)
            return checkpoint_path
        
        def save_last_checkpoint(self, checkpoint_path) -> str:
            with open(Path(checkpoint_path) / "last_checkpoint", "wb") as f:
                pickle.dump(self.__getstate__(), f)
            return checkpoint_path

        def load_checkpoint(self, checkpoint_path):
            print(f"Loading checkpoint at path {checkpoint_path}")
            checkpoint_file = list(Path(checkpoint_path).glob("checkpoint-*"))
            if len(checkpoint_file) == 0:
                raise RuntimeError("Missing checkpoint file!")
            RLTrainer.load_checkpoint(self, checkpoint_file[0])

    return MyTrainer


def get_config_from_file(filename: str, config: str):
    exec(open(filename).read(), globals())
    config = CONFIG[config]
    return config


parser = argparse.ArgumentParser()

parser.add_argument("--cluster", action="store_true")
parser.add_argument("--config", type=str, default="ppo")
parser.add_argument("--config-file", type=str, default="ray_config.py")
parser.add_argument("-n", "--name", type=str)
parser.add_argument("--env", type=str, default="data/env.xml")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--rollout", action="store_true")
parser.add_argument("--muscle_rollout", action="store_true")

if __name__ == "__main__":
    env_path = None
    checkpoint_path = None
    args = parser.parse_args()
    print("Argument : ", args)

    env_xml = Path(args.env).resolve()

    if args.cluster:
        ray.init(address=os.environ["ip_head"])
    else:
        if "node" in args.config:
            ray.init(num_cpus=128)
        else:
            ray.init()

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    config = get_config_from_file(args.config_file, args.config)
    ModelCatalog.register_custom_model("MyModel", SimulationNN_Ray)

    register_env("MyEnv", lambda config: MyEnv(env_xml))
    print(f"Loading config {args.config} from config file {args.config_file}.")

    config["rollout_fragment_length"] = config["train_batch_size"] / (
        config["num_workers"] * config["num_envs_per_worker"]
    )

    local_dir = "./ray_results"
    algorithm = config["trainer_config"]["algorithm"]
    MyTrainer = create_my_trainer(algorithm)

    from ray.tune import CLIReporter

    tune.run(
        MyTrainer,
        name=args.name,
        config=config,
        local_dir=local_dir,
        restore=checkpoint_path,
        progress_reporter=CLIReporter(max_report_frequency=25),
        checkpoint_freq=200,
    )

    ray.shutdown()
