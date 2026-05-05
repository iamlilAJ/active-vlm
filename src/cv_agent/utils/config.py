from langgraph.pregel.protocol import RunnableConfig
from omegaconf import DictConfig, OmegaConf

from cv_agent.core.registries import callback_handler_registry


def get_invocation_config(config: DictConfig) -> RunnableConfig:
    result = {}

    if "invocation_config" in config:
        inv_config = OmegaConf.to_container(config.invocation_config)
        assert isinstance(inv_config, dict)

        for key, val in inv_config.items():
            if key == "callbacks":
                result[key] = [callback_handler_registry.get(name) for name in val]
            else:
                result[key] = val

    return RunnableConfig(**result)
