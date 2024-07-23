"""Kaggle LLM 20 question competition agents.

https://www.kaggle.com/competitions/llm-20-questions
"""

import enum
import dataclasses
import os
import sys
import typing
import re
import random

import torch
import transformers

KAGGLE_SIMULATION_LIB_PATH = "/kaggle_simulations/agent/lib"
KAGGLE_SUBMISSION_LIB_PATH = "/kaggle/working/submission/lib"

if os.path.exists(KAGGLE_SIMULATION_LIB_PATH):
    sys.path.insert(0, KAGGLE_SIMULATION_LIB_PATH)
    MODEL_ROOT = "/kaggle/input/gemma/transformers"
elif os.path.exists(KAGGLE_SUBMISSION_LIB_PATH):
    sys.path.insert(0, KAGGLE_SUBMISSION_LIB_PATH)
    MODEL_ROOT = "/kaggle_simulations/agent/gemma/transformers"
else:
    MODEL_ROOT = "../gemma/models/transformers"

session_agent = None
session_model = None
session_tokenizer = None


class Role(enum.Enum):
    guesser = "guesser"
    answerer = "answerer"


@dataclasses.dataclass
class Question:
    prefix: str
    afirmative: str
    negative: str


class PromptBuilder:
    """Prompt builder for 20 questions game."""

    def __init__(
        self,
        user_chat_template: str,
        model_chat_template: str,
        model_chat_start: str,
        guess_prompt_prefix: typing.Optional[str] = "Name something that",
        guess_prompt_suffix: typing.Optional[str] = "The answer is",
        ask_fewshots: typing.Optional[list[str]] = None,
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
        self._ask_fewshots = (
            self._interleave_dialog(ask_fewshots) if ask_fewshots else ""
        )
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
        self._q_prefixes = [q.prefix for q in self._questions]

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
        self._qtype_pattern = re.compile(
            "|".join([q.prefix for q in self._questions])
        )

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
        self, questions: list[str], answers: list[str]
    ) -> str:
        if len(questions) != len(answers):
            raise ValueError(
                f"Got different number of {questions=} and {answers=}"
            )
        description = ""
        sep = ""
        for question, answer in zip(questions, answers):
            answer = self.clean_answer(answer)
            if answer not in ["yes", "no"]:
                continue
            question = self.clean_question(question)
            prefix = self._qtype_pattern.match(question)
            if not prefix:
                raise ValueError("Invalid question prefix")
            q_index = self._q_prefixes.index(prefix.group(0))
            if answer.lower() == "yes":
                verb = self._questions[q_index].afirmative
            else:
                verb = self._questions[q_index].negative
            suffix = question[len(self._questions[q_index].prefix) : -1]
            description += sep + verb + suffix
            sep = ", "

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

    def ask(
        self,
        questions: list[str],
        answers: list[str],
        random_seed: typing.Optional[int] = None,
    ) -> str:
        description = (
            "Ask a question about something that "
            + self.build_description(questions, answers)
        )
        random.seed(random_seed)
        question_prefix = random.choice(self._questions).prefix
        prompt = (
            self._ask_fewshots
            + self._user_chat_template.format(prompt=description)
            + self._model_chat_start
            + question_prefix
        )
        return prompt

    def guess(self, questions: list[str], answers: list[str]) -> str:
        description = self.build_description(questions, answers)
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
        question = re.sub(
            rf"\W({'|'.join(self._it_words)})\W",
            " " + keyword + " ",
            question,
            flags=re.IGNORECASE,
        )
        prompt = (
            self._user_chat_template.format(prompt="Yes or no." + question)
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
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._prompt_builder = prompt_builder

        self._answer_tokens = sorted(
            [
                token_id
                for token, token_id in self._tokenizer.vocab.items()
                if token.lower() in ["yes", "no"]
            ]
        )

    @torch.no_grad()
    def answer(self, observation) -> str:
        tokenized_prompt = self._tokenizer(
            self._prompt_builder.answer(
                observation.keyword, observation.questions[-1]
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
    def _generate(self, prompt: str, max_new_tokens: int = 3) -> str:
        tokenized_prompt = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        num_prompt_tokens = tokenized_prompt["input_ids"].shape[1]
        generated_tokens = self._model.generate(
            **tokenized_prompt, max_new_tokens=max_new_tokens
        )[0]

        question = self._tokenizer.decode(
            generated_tokens[max(0, num_prompt_tokens - 10) :]
        )
        model_chat_start = self._prompt_builder.get_model_chat_start_pattern()
        question = question[
            question.rfind(model_chat_start) + len(model_chat_start) :
        ]
        question = self._prompt_builder.clean_question(question)
        return question

    @torch.no_grad()
    def ask(self, observation) -> str:
        if observation.questions:
            question = self._generate(
                self._prompt_builder.ask(
                    observation.questions, observation.answers
                )
            )
        else:
            question = "Is it a place?"

        return question

    @torch.no_grad()
    def guess(self, observation) -> str:
        question = self._generate(
            self._prompt_builder.guess(
                observation.questions, observation.answers
            )
        )
        guess_suffix = self._prompt_builder.get_guess_prompt_suffix()
        guessed_word = question[question.rfind(guess_suffix) + len(guess_suffix) :]
        question = self._prompt_builder.clean_question("Is it" + guessed_word + "?")
        return question

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


def agent_fn(observation, *args, **kwargs):  # pylint: disable=unused-argument
    global session_agent, session_model, session_tokenizer
    if session_agent is None:
        variant = "7b-it"
        version = 3
        quantized = False
        device = torch.device("cpu")
        session_model, session_tokenizer = load_gemma_model_and_tokenizer(
            os.path.join(MODEL_ROOT, variant, str(version)),
            device=device,
            quantized=quantized,
        )
        session_agent = build_gemma_agent(session_model, session_tokenizer)
    response = session_agent(observation)
    return response
