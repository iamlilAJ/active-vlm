from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from cv_agent.core.builder import GraphBuilder
from cv_agent.utils.config import get_invocation_config

PUBLIC_CONFIGS = [
    Path("configs/boed-full.yaml"),
    Path("configs/lookahead-mme-remote-sensing.yaml"),
    Path("configs/mcmc-mme-remote-sensing.yaml"),
]


@pytest.fixture(autouse=True)
def public_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CV_AGENT_CROPPING_MCP_URL", "http://example.test/mcp")
    monkeypatch.setenv("MME_DATA_PATH", "/tmp/mme.json")
    monkeypatch.setenv("MME_IMAGE_DIR", "/tmp/mme-images")
    monkeypatch.setenv("VSTAR_DATA_PATH", "/tmp/vstar.jsonl")
    monkeypatch.setenv("VSTAR_IMAGE_DIR", "/tmp/vstar-images")


@pytest.mark.parametrize("config_path", PUBLIC_CONFIGS)
def test_public_configs_build_graph(config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    graph = GraphBuilder(cfg).build()
    invoke_config = get_invocation_config(cfg)

    assert graph is not None
    assert invoke_config is not None


def test_boed_full_is_primary_coverage_config() -> None:
    cfg = OmegaConf.load("configs/boed-full.yaml")

    assert set(cfg.benchmarks.keys()) == {"vstar", "cvbench", "hr4k", "hr8k", "mme"}
    assert cfg.workflow.nodes.action.name == "boed_crop_tool_executor"


def test_public_configs_do_not_embed_private_endpoints() -> None:
    private_ip_prefix = "45" + ".120."
    dummy_api_key = "api_key: " + "any"
    for config_path in PUBLIC_CONFIGS:
        text = config_path.read_text()
        assert private_ip_prefix not in text
        assert dummy_api_key not in text
        assert "aliyun" not in text.lower()
