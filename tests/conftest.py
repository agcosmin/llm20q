"""Test fixtures"""

import pytest

import llm20q.agents


@pytest.fixture(scope="session")
def chat_templates() -> tuple[str, str, str, list[str], list[str], list[str]]:
    user = "<start_turn>user\n{prompt}<end_turn>"
    model = "<start_turn>model\n{prompt}<end_turn>"
    model_start = "<start_turn>model\n"
    ask_shots = ["User ask", "Model response ask"]
    guess_shots = ["User guess", "Model response guess"]
    answer_shots = ["User answer", "Model response answer"]
    return user, model, model_start, ask_shots, guess_shots, answer_shots


@pytest.fixture(scope="session")
def prompt_builder(chat_templates) -> llm20q.agents.PromptBuilder:  # pylint: disable=redefined-outer-name
    user, model, model_start, ask_shots, guess_shots, answer_shots = (
        chat_templates
    )
    return llm20q.agents.PromptBuilder(
        user_chat_template=user,
        model_chat_template=model,
        model_chat_start=model_start,
        ask_fewshots=ask_shots,
        answer_fewshots=answer_shots,
        guess_fewshots=guess_shots,
    )
