import copy

CONFIG = dict()

common_config = {
    "env": "MyEnv",
    "trainer_config": {},
    "env_config": {},
    "framework": "torch",
    "extra_python_environs_for_driver": {},
    "extra_python_environs_for_worker": {},
    "model": {
        "custom_model": "MyModel",
        "custom_model_config": {
            "value_function": None
            },
            "max_seq_len": 0,
    },
    "evaluation_config": {},
}


CONFIG["ppo"] = copy.deepcopy(common_config)
CONFIG["ppo"]["trainer_config"]["algorithm"] = "PPO"
CONFIG["ppo"].update(   # Insert a new key-value, can insert many objects at once
    {   
        "horizon": 1000,
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.99,
        "gamma": 0.99,
        "kl_coeff": 0.00,
        "shuffle_sequences": True,
        "num_sgd_iter": 3,
        "lr": 1e-4,
        "lr_schedule": None,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.000,
        "entropy_coeff_schedule": None,
        "clip_param": 0.2,
        "vf_clip_param": 100.0,
        "grad_clip": None,
        "kl_target": 0.01,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "normalize_actions": False,
        "clip_actions": True,

        "create_env_on_driver": False,
        "num_cpus_for_driver": 0,
        "num_gpus": 1,
        "num_gpus_per_worker": 0.0,
        "num_envs_per_worker": 2,
        "num_cpus_per_worker": 1,
    }
)

## Discriminator Configuration
CONFIG["ppo"]["trainer_config"]["discriminator_lr"] = 5e-5
CONFIG["ppo"]["trainer_config"]["discriminator_num_epochs"] = 3
CONFIG["ppo"]["trainer_config"]["discriminator_w_penalty"] = 5.0

# World Configuration
CONFIG["ppo"]["trainer_config"]["world_lr"] = 5e-5
CONFIG["ppo"]["trainer_config"]["world_num_epochs"] = 3


# Large Set (For Cluster)
CONFIG["ppo_large"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_large"]["train_batch_size"] = 8192 * 8 * 4
CONFIG["ppo_large"]["sgd_minibatch_size"] = 4096

CONFIG["ppo_large"]["trainer_config"]["discriminator_sgd_minibatch_size"] = 4096
CONFIG["ppo_large"]["trainer_config"]["discriminator_batch_size"] = CONFIG["ppo_large"]["train_batch_size"]
CONFIG["ppo_large"]["trainer_config"]["world_sgd_minibatch_size"] = CONFIG["ppo_large"]["sgd_minibatch_size"]


# Medium Set (For a node or a PC)
CONFIG["ppo_medium"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_medium"]["train_batch_size"] = 8192 * 4
CONFIG["ppo_medium"]["sgd_minibatch_size"] = 512

CONFIG["ppo_medium"]["trainer_config"]["discriminator_sgd_minibatch_size"] = 512
CONFIG["ppo_medium"]["trainer_config"]["discriminator_batch_size"] = 8192
CONFIG["ppo_medium"]["trainer_config"]["world_sgd_minibatch_size"] = CONFIG["ppo_medium"]["sgd_minibatch_size"]


# Small Set (For a node or a PC)
CONFIG["ppo_small"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_small"]["train_batch_size"] = 8192
CONFIG["ppo_small"]["sgd_minibatch_size"] = 512

CONFIG["ppo_small"]["trainer_config"]["discriminator_sgd_minibatch_size"] = 256
CONFIG["ppo_small"]["trainer_config"]["discriminator_batch_size"] = 4096
CONFIG["ppo_small"]["trainer_config"]["world_sgd_minibatch_size"] = 256


## Mini Configuration (For DEBUG)
CONFIG["ppo_mini"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_mini"]["train_batch_size"] = 128
CONFIG["ppo_mini"]["sgd_minibatch_size"] = 64

CONFIG["ppo_mini"]["trainer_config"]["discriminator_sgd_minibatch_size"] = 64
CONFIG["ppo_mini"]["trainer_config"]["discriminator_batch_size"] = CONFIG["ppo_mini"]["train_batch_size"]
CONFIG["ppo_mini"]["trainer_config"]["world_sgd_minibatch_size"] = CONFIG["ppo_mini"]["train_batch_size"]

CONFIG["ppo_mini"]["num_workers"] = 1

# ===============================Training Configuration For Various Devices=========================================

# Large Set
CONFIG["ppo_large_server"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_server"]["num_workers"] = 128 * 4

CONFIG["ppo_large_node"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_node"]["num_workers"] = 128

CONFIG["ppo_large_pc"] = copy.deepcopy(CONFIG["ppo_large"])
CONFIG["ppo_large_pc"]["num_workers"] = 32

# Medium Set
CONFIG["ppo_medium_server"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_server"]["num_workers"] = 128 * 4

CONFIG["ppo_medium_node"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_node"]["num_workers"] = 128

CONFIG["ppo_medium_pc"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_pc"]["num_workers"] = 32

# Small Set
CONFIG["ppo_small_server"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["ppo_small_server"]["num_workers"] = 128 * 4

CONFIG["ppo_small_node"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["ppo_small_node"]["num_workers"] = 128

CONFIG["ppo_small_pc"] = copy.deepcopy(CONFIG["ppo_small"])
CONFIG["ppo_small_pc"]["num_workers"] = 16
