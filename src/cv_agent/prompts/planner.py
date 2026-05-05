from pathlib import Path

from cv_agent.core.registries import PromptGenerator, prompt_generator_registry

DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "planner_react.j2"


@prompt_generator_registry.register("planner")
def get_planner_prompt_generator(template_path: Path = DEFAULT_TEMPLATE_PATH) -> PromptGenerator:
    content = Path(template_path).read_text().strip()

    def generator(state: dict) -> str:
        del state
        return content

    return generator
