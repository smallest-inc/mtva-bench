import hashlib
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, Template, TemplateError

ENVIRONMENT = Environment(
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class Prompt:
    path: Path
    text: str
    template: Template = field(repr=False, compare=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def render(self, **values: Any) -> str:
        try:
            return self.template.render(**values)
        except TemplateError as error:
            raise ValueError(f"{self.path}: {error}") from error


COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)


def template_hash(path: Path) -> str:
    """A template's hash without its {# comments #}, which never reach a model."""
    text = COMMENT.sub("", path.read_text(encoding="utf-8"))
    return hashlib.sha256(text.encode()).hexdigest()


@cache
def load_prompt(path: Path) -> Prompt:
    text = path.read_text(encoding="utf-8")
    try:
        template = ENVIRONMENT.from_string(text)
    except TemplateError as error:
        raise ValueError(f"{path}: {error}") from error
    return Prompt(path=path, text=text, template=template)
