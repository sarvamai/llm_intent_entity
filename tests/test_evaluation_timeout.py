import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "src/llm_intent_entity/main.py"


def test_evaluation_client_uses_a_finite_timeout():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    timeout_seconds = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "EVALUATION_LLM_TIMEOUT_SECONDS":
                assert isinstance(node.value, ast.Constant)
                timeout_seconds = node.value.value
    assert isinstance(timeout_seconds, (int, float))
    assert timeout_seconds > 0

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "ChatCompletionsAPI":
            continue
        for keyword in node.keywords:
            if keyword.arg == "timeout":
                assert isinstance(keyword.value, ast.Name)
                assert keyword.value.id == "EVALUATION_LLM_TIMEOUT_SECONDS"
                found = True
    assert found
