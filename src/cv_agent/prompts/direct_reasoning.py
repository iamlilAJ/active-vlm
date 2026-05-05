from pathlib import Path

from jinja2 import Template

from cv_agent.core.registries import PromptGenerator, prompt_generator_registry

DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "direct_reasoning_system.j2"


@prompt_generator_registry.register("direct_reasoning")
def get_direct_reasoning_prompt_generator(
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> PromptGenerator:
    """Loads the prompt for the direct reasoning node."""
    template = Template(template_path.read_text())

    def generator(state: dict) -> str:
        if "question" in state:
            question = state["question"]
        else:
            raise ValueError(f"Cannot find question in state '{repr(state)}'.")

        return template.render(question=question).strip()

    return generator
