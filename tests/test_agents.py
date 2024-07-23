"""llm20q.agents tests"""

import re
import pytest

import llm20q.agents


def test_construct_prompt_bilder_without_prompt_pattern_raises_ex(
    chat_templates,
):
    user, model, model_start, *_ = chat_templates
    with pytest.raises(ValueError):
        llm20q.agents.PromptBuilder("user_prompt", model, model_start)
    with pytest.raises(ValueError):
        llm20q.agents.PromptBuilder(user, "model_prompt", model_start)


def test_build_description_with_no_questions_and_answers_results_in_empty_str(
    prompt_builder,
):
    assert prompt_builder.build_description([], []) == ""


def test_build_desc_raises_for_mismatched_question_and_answers_length(
    prompt_builder,
):
    with pytest.raises(ValueError):
        prompt_builder.build_description(["Is it a cow"], [])
    with pytest.raises(ValueError):
        prompt_builder.build_description([], ["Yes", "no"])


def test_build_description_with_invalid_question_raises_ex(prompt_builder):
    with pytest.raises(ValueError):
        prompt_builder.build_description(["How many coins?"], ["yes"])


def test_build_description_give_exepcted_results_for_all_question_types(
    prompt_builder,
):
    samples = [
        ("Is it", "an animal?", "is an animal", "is not an animal"),
        ("Does it", "bark?", "does bark", "does not bark"),
        ("Does it have", "a tail?", "does have a tail", "does not have a tail"),
    ]

    for prefix, suffix, affirmative, negative in samples:
        question = prefix + " " + suffix
        assert affirmative == prompt_builder.build_description(
            [question], ["yes"]
        )
        assert negative == prompt_builder.build_description([question], ["no"])


def test_build_description_from_question_with_invalid_chars(prompt_builder):
    question = ["Is it a*(big     *country?**?---?"]
    assert "is a big country" == prompt_builder.build_description(
        question, ["yes"]
    )
    assert "is not a big country" == prompt_builder.build_description(
        question, ["no"]
    )


def test_build_chained_description(prompt_builder):
    desc = prompt_builder.build_description(
        ["Is it an animal?", "Does it bark?", "Does it have a tail?"],
        ["yes", "no", "yes"],
    )
    assert "is an animal, does not bark, does have a tail" == desc


def test_build_description_with_non_yes_or_no_answer(prompt_builder):
    assert "" == prompt_builder.build_description(
        ["Is it an animal?"], ["yes no"]
    )
    assert "does bark" == prompt_builder.build_description(
        ["Is it an animal?", "Does it bark?"], ["something someting***", "yes"]
    )


@pytest.mark.parametrize(
    "answer", ["Yes", "yEs", "* yeS", "   YES  ()?", "ye_s"]
)
def test_build_description_with_non_standard_yes_answer(prompt_builder, answer):
    assert "is an animal" == prompt_builder.build_description(
        ["Is it an animal?"], [answer]
    )


@pytest.mark.parametrize("answer", ["No", "nO", "* NO", "   NO  ()?", "n-o"])
def test_build_description_with_non_standard_no_answer(prompt_builder, answer):
    assert "is not an animal" == prompt_builder.build_description(
        ["Is it an animal?"], [answer]
    )


def test_interleave_dialog(prompt_builder):
    # pylint: disable=protected-access
    user_first_lines = ["User line", "Model line", "User line"]
    user_first_dialog = prompt_builder._interleave_dialog(
        user_first_lines, model_first=False
    )
    user_first_expected_dialog = (
        prompt_builder._user_chat_template.format(prompt=user_first_lines[0])
        + prompt_builder._model_chat_template.format(prompt=user_first_lines[1])
        + prompt_builder._user_chat_template.format(prompt=user_first_lines[2])
    )
    assert user_first_expected_dialog == user_first_dialog

    model_first_lines = ["Model line", "User line", "Model line"]
    model_first_dialog = prompt_builder._interleave_dialog(
        model_first_lines, model_first=True
    )
    model_first_expected_dialog = (
        prompt_builder._model_chat_template.format(prompt=model_first_lines[0])
        + prompt_builder._user_chat_template.format(prompt=model_first_lines[1])
        + prompt_builder._model_chat_template.format(
            prompt=model_first_lines[2]
        )
    )
    assert model_first_expected_dialog == model_first_dialog


def test_interleave_dialog_with_custom_separator(prompt_builder):
    # pylint: disable=protected-access
    lines = ["User line", "Model line", "User line"]
    sep = "@@@"
    dialog = prompt_builder._interleave_dialog(
        lines, sep=sep, model_first=False
    )
    expected_dialog = (
        prompt_builder._user_chat_template.format(prompt=lines[0])
        + sep
        + prompt_builder._model_chat_template.format(prompt=lines[1])
        + sep
        + prompt_builder._user_chat_template.format(prompt=lines[2])
    )
    assert expected_dialog == dialog


def test_generate_ask_prompt_with_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="<bos>user:{prompt}<eos>",
        model_chat_template="<bos>model:{prompt}<eos>",
        model_chat_start="<bos>model:",
        ask_fewshots=[
            "You must guess someting by asking questions about it."
            + "Ask question about something that is a place, is in Europe",
            "Is it near France?",
            "Ask question about something that is not a place, does bark",
            "Does it have a tail?",
        ],
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "<bos>user:You must guess someting by asking questions about it."
        + "Ask question about something that is a place, is in Europe<eos>"
        + r"<bos>model:Is it near France\?<eos>"
        + "<bos>user:Ask question about something that is not a place, does bark<eos>"  # pylint: disable=line-too-long
        + r"<bos>model:Does it have a tail\?<eos>"
        + "<bos>user:Ask a question about something that is an animal, does not meow<eos>"  # pylint: disable=line-too-long
        + "<bos>model:(Is it|Does it have|Does it)"
    )
    prompt = builder.ask(questions, answers)
    assert re.fullmatch(expected_prompt, prompt)


def test_generate_ask_prompt_without_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="<bos>user:{prompt}<eos>",
        model_chat_template="<bos>model:{prompt}<eos>",
        model_chat_start="<bos>model:",
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "<bos>user:Ask a question about something that is an animal, does not meow<eos>"  # pylint: disable=line-too-long
        + "<bos>model:(Is it|Does it have|Does it)"
    )
    prompt = builder.ask(questions, answers)
    assert re.fullmatch(expected_prompt, prompt)


def test_generate_guess_prompt_with_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="<bos>user:{prompt}<eos>",
        model_chat_template="<bos>model:{prompt}<eos>",
        model_chat_start="<bos>model:",
        guess_fewshots=[
            "You must name someting based on the following description."
            + "Name something that is a place, is in Europe",
            "The answer is France",
            "Name something that is not a place, does bark",
            "The answer is a dog",
        ],
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "<bos>user:You must name someting based on the following description."
        + "Name something that is a place, is in Europe<eos>"
        + r"<bos>model:The answer is France<eos>"
        + "<bos>user:Name something that is not a place, does bark<eos>"
        + r"<bos>model:The answer is a dog<eos>"
        + "<bos>user:Name something that is an animal, does not meow<eos>"
        + "<bos>model:The answer is"
    )
    prompt = builder.guess(questions, answers)
    assert re.fullmatch(expected_prompt, prompt)


def test_generate_guess_prompt_without_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="<bos>user:{prompt}<eos>",
        model_chat_template="<bos>model:{prompt}<eos>",
        model_chat_start="<bos>model:",
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "<bos>user:Name something that is an animal, does not meow<eos>"
        + "<bos>model:The answer is"
    )
    prompt = builder.guess(questions, answers)
    assert re.fullmatch(expected_prompt, prompt)


def test_generate_guess_prompt_with_custom_prefix_and_suffix():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="<bos>user:{prompt}<eos>",
        model_chat_template="<bos>model:{prompt}<eos>",
        model_chat_start="<bos>model:",
        guess_prompt_prefix="What",
        guess_prompt_suffix="The answer to the question is",
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "<bos>user:What is an animal, does not meow<eos>"
        + "<bos>model:The answer to the question is"
    )
    prompt = builder.guess(questions, answers)
    assert re.fullmatch(expected_prompt, prompt)


@pytest.mark.parametrize(
    "question",
    [
        "Does {it_word} have a tail?",
        "Is the {it_word} a country?",
        "Does {it_word} bark?",
    ],
)
@pytest.mark.parametrize(
    "it_word", ["it", "secret word", "secret", "hidden word", "keyword"]
)
def test_generate_answer_prompt(question, it_word):
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="<bos>user:{prompt}<eos>",
        model_chat_template="<bos>model:{prompt}<eos>",
        model_chat_start="<bos>model:",
    )
    keyword = "dog"
    prompt = builder.answer(keyword, question.format(it_word=it_word))
    expected_prompt = (
        "<bos>user:Yes or no."
        + question.format(it_word=keyword)
        + "<eos>"
        + "<bos>model:"
    )
    assert expected_prompt == prompt


def test_generate_answer_prompt_with_dirty_question():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="<bos>user:{prompt}<eos>",
        model_chat_template="<bos>model:{prompt}<eos>",
        model_chat_start="<bos>model:",
    )
    keyword = "dog"
    question = "Does+the,+ it+, have a hat???"
    prompt = builder.answer(keyword, question)
    expected_prompt = (
        "<bos>user:Yes or no.Does the dog have a hat?<eos>" + "<bos>model:"
    )
    assert expected_prompt == prompt
