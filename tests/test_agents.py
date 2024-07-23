"""llm20q.agents tests"""

import re
import pytest

import llm20q.agents


def test_construct_prompt_bilder_without_prompt_pattern_raises_ex():
    with pytest.raises(ValueError):
        llm20q.agents.PromptBuilder(
            user_chat_template="user_bad_pattern",
            model_chat_template="model:{prompt}\n",
            model_chat_start="model:",
        )
    with pytest.raises(ValueError):
        llm20q.agents.PromptBuilder(
            user_chat_template="user:{prompt}",
            model_chat_template="model_bad_pattern",
            model_chat_start="model:",
        )


def test_build_description_with_no_questions_and_answers_results_in_empty_str():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    assert builder.build_description([], []) == ""


def test_build_desc_raises_for_mismatched_question_and_answers_length():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    with pytest.raises(ValueError):
        builder.build_description(["Is it a cow"], [])
    with pytest.raises(ValueError):
        builder.build_description([], ["Yes", "no"])


def test_build_description_with_invalid_question_raises_ex():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    with pytest.raises(ValueError):
        builder.build_description(["How many coins?"], ["yes"])


def test_build_description_give_exepcted_results_for_all_question_types():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    samples = [
        ("Is it", "an animal?", "is an animal", "is not an animal"),
        ("Does it", "bark?", "does bark", "does not bark"),
        ("Does it have", "a tail?", "does have a tail", "does not have a tail"),
    ]

    for prefix, suffix, affirmative, negative in samples:
        question = prefix + " " + suffix
        assert builder.build_description([question], ["yes"]) == affirmative
        assert builder.build_description([question], ["no"]) == negative


def test_build_description_from_question_with_invalid_chars():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    question = ["Is it a*(big     *country?**?---?"]
    assert builder.build_description(question, ["yes"]) == "is a big country"
    assert builder.build_description(question, ["no"]) == "is not a big country"


def test_build_chained_description():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    desc = builder.build_description(
        ["Is it an animal?", "Does it bark?", "Does it have a tail?"],
        ["yes", "no", "yes"],
    )
    assert desc == "is an animal, does not bark, does have a tail"


def test_build_description_with_non_yes_or_no_answer():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    assert builder.build_description(["Is it an animal?"], ["yes no"]) == ""
    one_bad_one_good_answer_prompt = builder.build_description(
        ["Is it an animal?", "Does it bark?"], ["something someting***", "yes"]
    )
    assert one_bad_one_good_answer_prompt == "does bark"


@pytest.mark.parametrize(
    "answer", ["Yes", "yEs", "* yeS", "   YES  ()?", "ye_s"]
)
def test_build_description_with_non_standard_yes_answer(answer):
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    assert builder.build_description(["Is it a dog?"], [answer]) == "is a dog"


@pytest.mark.parametrize("answer", ["No", "nO", "* NO", "   NO  ()?", "n-o"])
def test_build_description_with_non_standard_no_answer(answer):
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    assert (
        builder.build_description(["Is it a dog?"], [answer]) == "is not a dog"
    )


def test_interleave_dialog():
    # pylint: disable=protected-access
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    user_first_lines = ["User line", "Model line", "User line"]
    user_first_dialog = builder._interleave_dialog(
        user_first_lines, model_first=False
    )
    user_first_expected_dialog = (
        "user:User line\n" + "model:Model line\n" + "user:User line\n"
    )
    assert user_first_dialog == user_first_expected_dialog

    model_first_lines = ["Model line", "User line", "Model line"]
    model_first_dialog = builder._interleave_dialog(
        model_first_lines, model_first=True
    )
    model_first_expected_dialog = (
        "model:Model line\n" + "user:User line\n" + "model:Model line\n"
    )
    assert model_first_dialog == model_first_expected_dialog


def test_interleave_dialog_with_custom_separator():
    # pylint: disable=protected-access
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    lines = ["User line", "Model line", "User line"]
    sep = "@@@"
    dialog = builder._interleave_dialog(lines, sep=sep, model_first=False)
    expected_dialog = (
        "user:User line\n@@@" + "model:Model line\n@@@" + "user:User line\n"
    )
    assert dialog == expected_dialog


def test_generate_ask_prompt_with_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
        ask_prompt_suffix="Ask a question:",
        ask_fewshots=[
            "Describe ask action.User ask example 1",
            "Model ask answer 1",
            "User ask example 2",
            "Model ask answer 2",
        ],
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "user:Describe ask action.User ask example 1\n"
        + "model:Model ask answer 1\n"
        + "user:User ask example 2\n"
        + "model:Model ask answer 2\n"
        + "user:Ask a question: is an animal, does not meow\n"
        + "model:(Is it|Does it have|Does it)"
    )
    prompt = builder.ask(questions, answers)
    assert re.fullmatch(expected_prompt, prompt)


def test_generate_ask_prompt_without_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
        ask_prompt_suffix="Ask a question:",
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "user:Ask a question: is an animal, does not meow\n"
        + "model:(Is it|Does it have|Does it)"
    )
    prompt = builder.ask(questions, answers)
    assert re.fullmatch(expected_prompt, prompt)


def test_generate_guess_prompt_with_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
        guess_prompt_prefix="Name something that",
        guess_prompt_suffix="The answer is",
        guess_fewshots=[
            "Describe guess action. User guess example 1",
            "Model guess example 1",
            "User guess example 2",
            "Model guess example 2",
        ],
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "user:Describe guess action. User guess example 1\n"
        + "model:Model guess example 1\n"
        + "user:User guess example 2\n"
        + "model:Model guess example 2\n"
        + "user:Name something that is an animal, does not meow\n"
        + "model:The answer is"
    )
    prompt = builder.guess(questions, answers)
    assert prompt == expected_prompt


def test_generate_guess_prompt_without_fewshots():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
        guess_prompt_prefix="Name something that",
        guess_prompt_suffix="The answer is",
    )
    questions = ["Is it an animal?", "Does it meow?"]
    answers = ["yes", "no"]
    expected_prompt = (
        "user:Name something that is an animal, does not meow\n"
        + "model:The answer is"
    )
    prompt = builder.guess(questions, answers)
    assert prompt == expected_prompt


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
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    keyword = "dog"
    prompt = builder.answer(keyword, question.format(it_word=it_word))
    expected_prompt = (
        f"user:Yes or no.{question.format(it_word=keyword)}\nmodel:"
    )
    assert expected_prompt == prompt


def test_generate_answer_prompt_with_dirty_question():
    builder = llm20q.agents.PromptBuilder(
        user_chat_template="user:{prompt}\n",
        model_chat_template="model:{prompt}\n",
        model_chat_start="model:",
    )
    keyword = "dog"
    question = "Does+the,+ it+, have a hat???"
    prompt = builder.answer(keyword, question)
    expected_prompt = "user:Yes or no.Does the dog have a hat?\nmodel:"
    assert expected_prompt == prompt
