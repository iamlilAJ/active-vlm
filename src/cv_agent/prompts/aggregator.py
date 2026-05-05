from pathlib import Path

from jinja2 import Template

from cv_agent.core.registries import PromptGenerator, prompt_generator_registry

DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "aggregator_system.j2"


@prompt_generator_registry.register("aggregator_system")
def get_aggregator_prompt_generator(template_path: Path = DEFAULT_TEMPLATE_PATH) -> PromptGenerator:
    """Loads the system prompt for the final aggregator node."""
    template = Template(template_path.read_text())

    def generator(state: dict) -> str:
        question = state.get("question")
        if not question:
            raise ValueError(f"Cannot find question in state '{repr(state)}'.")

        return template.render(question=question).strip()

    return generator
