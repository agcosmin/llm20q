"""Kaggle LLM 20 question competition agents.

https://www.kaggle.com/competitions/llm-20-questions
"""

import dataclasses
import os
import os.path
import sys
import typing
import re
import random

import torch
import transformers

KAGGLE_SIMULATION_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_SIMULATION_AGENT_PATH):
    sys.path.insert(0, os.path.join(KAGGLE_SIMULATION_AGENT_PATH, "lib"))
    MODEL_ROOT = os.path.join(KAGGLE_SIMULATION_AGENT_PATH, "transformers")
else:
    MODEL_ROOT = "./gemma/models/transformers"

session_agents = {}
session_models = {}
session_tokenizers = {}
session_bad_words_ids = {}
session_prompt_builders = {}


@dataclasses.dataclass
class Question:
    prefix: str
    afirmative: str
    negative: str


class QuestionSelector:
    """Simple logic tree to select questions to determine basic info."""

    def __init__(self) -> None:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        group_size = 4
        start_letter_question = (
            "Does it's name start with one of these letters: {g}?"
        )
        start_letter = {
            start_letter_question.format(
                g=",".join(
                    list(alphabet[i : min(i + group_size, len(alphabet))])
                )
            ): None
            for i in range(0, len(alphabet), group_size)
        }
        start_letter["default"] = None
        continent = {
            f"Is it from {x}?": start_letter
            for x in [
                "Europe",
                "Africa",
                "North America",
                "South America",
                "Australia",
                "Asia",
            ]
        }
        continent["default"] = start_letter
        place_kind = {
            f"Is it a {x}?": continent
            for x in [
                "country",
                "city",
                "mountain",
                "river",
                "lake",
                "sea",
                "ocean",
            ]
        }
        place_kind["default"] = continent
        thing_kind = {
            f"Is it a {x}?": start_letter
            for x in ["tool", "device", "vehicle", "animal"]
        }
        thing_kind["default"] = start_letter
        domain = {
            f"Is it used in {x}?": thing_kind
            for x in [
                "cooking",
                "transport",
                "entertainment",
                "construction",
                "carpentry",
                "surgery",
                "clothing",
                "writing",
                "reading",
                "cleaning",
                "food",
                "drink",
            ]
        }
        domain["default"] = thing_kind
        self._question_graph = {"Is it a place?": place_kind, "default": domain}

    def select_question(self, questions: list[str], answers: list[str]) -> str:
        current = self._question_graph
        used = set()
        for question, answer in zip(questions, answers):
            if answer == "yes":
                current = current[question]
                used = set()
            else:
                used.add(question)
                if len(used) == len(current) - 1:
                    # All Q from this level are answered with no
                    current = current["default"]
                    used = set()
            if not current:
                break

        if current:
            used.add("default")
            options = list(set(current.keys()) - used)
            next_question = random.choice(options)
            return next_question
        else:
            return None


class PromptBuilder:
    """Prompt builder for 20 questions game."""

    def __init__(
        self,
        user_chat_template: str,
        model_chat_template: str,
        model_chat_start: str,
        guess_prompt_prefix: str = "",
        guess_prompt_suffix: str = "",
        guess_fewshots: typing.Optional[list[str]] = None,
        answer_fewshots: typing.Optional[list[str]] = None,
        questions: typing.Optional[list[str]] = None,
        it_words: typing.Optional[list[str]] = None,
    ) -> None:
        prompt_pattern = "{prompt}"
        if (
            user_chat_template.find(prompt_pattern) < 0
            or model_chat_template.find(prompt_pattern) < 0
        ):
            raise ValueError(
                f"Chat templates must contain the {prompt_pattern}"
            )

        self._user_chat_template = user_chat_template
        self._model_chat_template = model_chat_template
        self._model_chat_start = model_chat_start
        self._guess_fewshots = (
            self._interleave_dialog(guess_fewshots) if guess_fewshots else ""
        )
        self._answer_fewshots = (
            self._interleave_dialog(answer_fewshots) if answer_fewshots else ""
        )
        self._guess_prompt_prefix = guess_prompt_prefix
        self._guess_prompt_suffix = guess_prompt_suffix

        if not questions:
            questions = [
                Question("Is it", "is", "is not"),
                Question("Does it have", "does have", "does not have"),
                Question("Does it", "does", "does not"),
            ]

        self._questions = sorted(
            questions, key=lambda q: q.prefix, reverse=True
        )

        if it_words:
            self._it_words = sorted(it_words, reverse=True)
        else:
            self._it_words = [
                "it",
                "secret word",
                "secret",
                "hidden word",
                "keyword",
            ]

    @staticmethod
    def clean_question(question: str) -> str:
        question = re.sub(r"[^a-zA-z0-9 _-]", " ", question)
        question = re.sub(r"[\s]{2,}", " ", question)
        question = re.sub("[^a-zA-Z0-9]+$", "?", question)
        if question[-1] != "?":
            question += "?"
        return question

    @staticmethod
    def clean_answer(answer: str) -> str:
        answer = re.sub(r"[^a-zA-Z]", "", answer).lower()
        return answer

    def get_model_chat_start_pattern(self) -> str:
        return self._model_chat_start

    def get_guess_prompt_suffix(self) -> str:
        return self._guess_prompt_suffix

    def build_description(
        self,
        questions: list[str],
        answers: list[str],
        guesses: list[str],
        only_positive: bool = False,
    ) -> str:
        if len(questions) != len(answers):
            raise ValueError(
                f"Got different number of {questions=} and {answers=}"
            )
        description = ""
        sep = ""
        for question, answer in zip(questions, answers):
            answer = self.clean_answer(answer)
            use_answers = ["yes"] + ([] if only_positive else ["no"])
            if answer not in use_answers:
                continue
            question = self.clean_question(question)
            q_index = None
            for i, q in enumerate(self._questions):
                if question[0 : len(q.prefix)] == q.prefix:
                    q_index = i
            if q_index is None:
                raise ValueError(f"Invalid question prefix: {question}")

            if answer.lower() == "yes":
                verb = self._questions[q_index].afirmative
            else:
                verb = self._questions[q_index].negative
            suffix = question[len(self._questions[q_index].prefix) : -1]
            description += sep + verb + suffix
            sep = ", "

        guesses = set(guess.lower() for guess in guesses)
        if guesses:
            description += " and is not one of: " + ", ".join(guesses)
        return description

    def _interleave_dialog(
        self, lines: list[str], sep: str = "", model_first: bool = False
    ) -> str:
        ping = self._user_chat_template
        pong = self._model_chat_template
        if model_first:
            ping, pong = pong, ping

        dialog = sep.join(
            [
                ping.format(prompt=line) if i % 2 else pong.format(prompt=line)
                for i, line in enumerate(lines, start=1)
            ]
        )
        return dialog

    def guess(
        self,
        questions: list[str],
        answers: list[str],
        guesses: list[str],
        only_positive: bool = True,
    ) -> str:
        description = self.build_description(
            questions, answers, guesses, only_positive=only_positive
        )
        prompt = (
            self._guess_fewshots
            + self._user_chat_template.format(
                prompt=self._guess_prompt_prefix + " " + description
            )
            + self._model_chat_start
            + self._guess_prompt_suffix
        )
        return prompt

    def answer(self, keyword: str, question: str) -> str:
        question = self.clean_question(question)
        keywords = keyword.lower().split(" ")
        random.shuffle(keywords)
        keyword = " ".join(keywords)
        question = re.sub(
            rf"\W({'|'.join(self._it_words)})\W",
            " " + keyword + " ",
            question,
            flags=re.IGNORECASE,
        )
        prompt = (
            self._user_chat_template.format(
                prompt="Answer the next question with no or yes only: "
                + question
            )
            + self._model_chat_start
        )
        return prompt


class LLMAgent:
    """20 Questions game LLM agent."""

    def __init__(
        self,
        model: typing.Any,
        tokenizer: typing.Any,
        prompt_builder: PromptBuilder,
        generation_config: typing.Optional[
            transformers.GenerationConfig
        ] = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._prompt_builder = prompt_builder
        self._generation_config = generation_config

        self._answer_tokens = sorted(
            [
                token_id
                for token, token_id in self._tokenizer.vocab.items()
                if token.lower() in ["yes", "no"]
            ]
        )
        self._question_selector = QuestionSelector()

    @torch.no_grad()
    def answer(self, observation) -> str:
        tokenized_prompt = self._tokenizer(
            self._prompt_builder.answer(
                observation.keyword, observation.questions[-1].lower()
            ),
            return_tensors="pt",
        ).to(self._model.device)
        logits = self._model(**tokenized_prompt, return_dict=True).logits
        next_token_logits = logits[:, -1, self._answer_tokens]
        answer = self._tokenizer.decode(
            self._answer_tokens[torch.argmax(next_token_logits, axis=-1).item()]
        ).lower()
        return answer

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 2) -> str:
        tokenized_prompt = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        num_prompt_tokens = tokenized_prompt["input_ids"].shape[1]
        generated_tokens = self._model.generate(
            **tokenized_prompt,
            max_new_tokens=max_new_tokens,
            generation_config=self._generation_config,
        )[0]

        question = self._tokenizer.decode(
            generated_tokens[max(0, num_prompt_tokens - 20) :]
        )
        model_chat_start = self._prompt_builder.get_model_chat_start_pattern()
        question = question[
            question.rfind(model_chat_start) + len(model_chat_start) :
        ]
        question = self._prompt_builder.clean_question(question)
        return question

    @torch.no_grad()
    def ask(self, observation) -> str:
        question = self._question_selector.select_question(
            observation.questions, observation.answers
        )
        if not question:
            guessed_word = self.guess(observation)
            question = self._prompt_builder.clean_question(
                f"Is it {guessed_word}?"
            )
        return question

    @torch.no_grad()
    def guess(self, observation) -> str:
        question = self._generate(
            self._prompt_builder.guess(
                observation.questions,
                observation.answers,
                observation.guesses,
                only_positive=True,
            ),
            max_new_tokens=5,
        )
        question = self._prompt_builder.clean_question(question)
        guess_suffix = self._prompt_builder.get_guess_prompt_suffix()
        guessed_words = question[
            question.rfind(guess_suffix) + len(guess_suffix) : -1
        ].strip()

        clean_guessed_words = ""
        for word in guessed_words.lower().split(" "):
            if len(word) > 3 and word not in clean_guessed_words:
                clean_guessed_words += word + " "
        clean_guessed_words = clean_guessed_words[0:-1]

        if len(clean_guessed_words) == 0:
            # must guess something otherwise the game errors out
            clean_guessed_words = "dog"

        return clean_guessed_words

    def __call__(self, observation, *args) -> str:
        if observation.turnType == "ask":
            return self.ask(observation)
        elif observation.turnType == "guess":
            return self.guess(observation)
        elif observation.turnType == "answer":
            return self.answer(observation)
        else:
            raise ValueError(f"Unknown {observation.turnType} turn type.")


def load_gemma_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
    quantized: bool = False,
) -> tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    quantized_cfg = {}
    if quantized:
        quantized_cfg = {
            "quantization_config": transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path, device_map={"": device}, **quantized_cfg
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def build_gemma_prompt_builder(
    guess_prompt_prefix: str,
    guess_prompt_suffix: str,
    guess_fewshots: typing.Optional[list[str]] = None,
    answer_fewshots: typing.Optional[list[str]] = None,
) -> PromptBuilder:
    return PromptBuilder(
        user_chat_template="<start_of_turn>user\n{prompt}<end_of_turn>\n",
        model_chat_template="<start_of_turn>model\n{prompt}<end_of_turn>\n",
        model_chat_start="<start_of_turn>model\n",
        guess_prompt_prefix=guess_prompt_prefix,
        guess_prompt_suffix=guess_prompt_suffix,
        guess_fewshots=guess_fewshots,
        answer_fewshots=answer_fewshots,
    )


def build_gemma_agent(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
) -> LLMAgent:
    return LLMAgent(
        model=model,
        tokenizer=tokenizer,
        prompt_builder=PromptBuilder(
            user_chat_template="<start_of_turn>user\n{prompt}<end_of_turn>\n",
            model_chat_template="<start_of_turn>model\n{prompt}<end_of_turn>\n",
            model_chat_start="<start_of_turn>model\n",
        ),
    )


def get_bad_tokens(words: list[str], tokenizer) -> list[list[int]]:
    bad_words_raw = [
        [token_id]
        for token, token_id in tokenizer.vocab.items()
        if any(word.lower() in token.lower() for word in words)
    ]
    bad_words_tokenized = [
        tokenizer([" " + word], add_special_tokens=False).input_ids[0]
        for word in words
    ]
    bad_words = bad_words_raw + [
        tokens for tokens in bad_words_tokenized if tokens not in bad_words_raw
    ]
    return bad_words


def agent_fn(observation, *args, **kwargs) -> str:  # pylint: disable=unused-argument
    agent_id = kwargs.get("llm20q_agent_id", "default")
    agent = session_agents.get(agent_id, None)
    if agent is None:
        model_id = kwargs.get("llm20q_model_id", "default")
        model = session_models.get(model_id, None)
        tokenizer = session_tokenizers.get(model_id, None)
        prompt_builder = session_prompt_builders.get(model_id, None)
        generation_config = None
        if model is None or tokenizer is None:
            if model_id == "default":
                variant = "2b-it"
                version = 3
                quantized = kwargs.get("llm20q_use_quantized_model", False)
                device = kwargs.get("llm20q_device", torch.device("cpu"))
                checkpoint_path = os.path.join(
                    MODEL_ROOT, variant, str(version)
                )
                model, tokenizer = load_gemma_model_and_tokenizer(
                    checkpoint_path,
                    device=device,
                    quantized=quantized,
                )
                session_models[model_id] = model
                session_tokenizers[model_id] = tokenizer
                prompt_builder = build_gemma_prompt_builder(
                    "Name something that", "The answer is"
                )

                generation_config = (
                    transformers.GenerationConfig.from_pretrained(
                        checkpoint_path,
                        num_beams=2,
                        do_sample=True,
                        temperature=5.0,
                    )
                )
            else:
                raise ValueError(f"Unknown model id: {model_id}")

        generation_config_args = kwargs.get("llm20q_generation_config_args", {})
        bad_words = kwargs.get(
            "llm20q_bad_words",
            [
                "none",
                "nothing",
                "somewhere",
                "anywhere",
                "here",
                "there",
                "word",
                "hypothetical",
                "mythical",
                "fiction",
                "name",
                "the",
            ],
        )

        if bad_words:
            generation_config_args["bad_words_ids"] = get_bad_tokens(
                bad_words, tokenizer
            )
        if not generation_config:
            generation_config = transformers.GenerationConfig.from_model_config(
                model.config
            )
        generation_config.update(**generation_config_args)

        agent = LLMAgent(
            model=model,
            tokenizer=tokenizer,
            prompt_builder=prompt_builder,
            generation_config=generation_config,
        )
        session_agents[agent_id] = agent

    response = agent(observation)
    return response
