"""Test fixtures"""

import pytest

import llm20q.agents


@pytest.fixture(scope="session")
def chat_templates() -> tuple[str, str, str, list[str], list[str], list[str]]:
    user = "<start_turn>user\n{prompt}<end_turn>"
    model = "<start_turn>model\n{prompt}<end_turn>"
    model_start = "<start_turn>model\n"
    guess_prefix = "Name something that"
    guess_suffix = "The answer is"
    ask_suffix = "Ask a question about a thing"
    ask_shots = ["User ask", "Model response ask"]
    guess_shots = ["User guess", "Model response guess"]
    answer_shots = ["User answer", "Model response answer"]
    args = (
        user,
        model,
        model_start,
        guess_prefix,
        guess_suffix,
        ask_suffix,
        ask_shots,
        guess_shots,
        answer_shots,
    )
    return args


@pytest.fixture(scope="session")
def prompt_builder(chat_templates) -> llm20q.agents.PromptBuilder:  # pylint: disable=redefined-outer-name
    return llm20q.agents.PromptBuilder(*chat_templates)
