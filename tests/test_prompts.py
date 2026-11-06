from pathlib import Path

import pytest

from mtva_bench.prompts import load_prompt


def test_render_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "judge.md"
    path.write_text(
        "Rules:\n{% for rule in rules %}\n- {{ rule }}\n{% endfor %}\nAgent: {{ agent }}\n"
    )
    prompt = load_prompt(path)
    assert prompt.render(rules=["Be brief", "Confirm"], agent="{{customer_name}}") == (
        "Rules:\n- Be brief\n- Confirm\nAgent: {{customer_name}}\n"
    )
    assert len(prompt.sha256) == 64
    assert load_prompt(path) is prompt


def test_missing_variable_names_the_template(tmp_path: Path) -> None:
    path = tmp_path / "caller.md"
    path.write_text("Persona: {{ persona }}")
    with pytest.raises(ValueError, match=r"caller\.md: 'persona' is undefined"):
        load_prompt(path).render()
