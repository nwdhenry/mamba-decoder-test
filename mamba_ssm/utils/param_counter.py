import argparse
from mamba_ssm.models.mamba_gpt import MambaGPT, MambaGPTConfig
from mamba_ssm.training.autoconfig import PRESET_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base")
    args = parser.parse_args()

    cfg_kwargs = PRESET_CONFIGS.get(args.config, {})
    config = MambaGPTConfig(**cfg_kwargs)
    model = MambaGPT(config)
    num_params = model.num_parameters(only_trainable=True)
    print(num_params)


if __name__ == "__main__":
    main()
