"""LLM20 play main application"""

import argparse
import typing

import kaggle_environments as kaggle_env
import torch

import llm20q.agents


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable environment debug prints",
    )
    return parser


class Agent:
    """LLM20 agent wrapper."""

    def __init__(
        self,
        agent_id: str = "default",
        model_id: bool = "default",
        quantized_model: bool = False,
        device: torch.device = torch.device("cpu"),
        bad_words: typing.Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        self._kwargs = {
            "llm20q_model_id": model_id,
            "llm20q_agent_id": agent_id,
            "llm20q_use_quantized_model": quantized_model,
            "llm20q_device": device,
            "llm20q_generation_config_args": kwargs,
        }
        if bad_words:
            self._kwargs["llm20q_bad_words"] = bad_words

    def __call__(self, observation, *args, **kwargs) -> str:
        response = llm20q.agents.agent_fn(
            observation, *args, **kwargs, **self._kwargs
        )
        return response


def play(debug: bool = False):
    env = kaggle_env.make("llm_20_questions", debug=debug)
    guesser_1 = Agent(quantized_model=True, device=torch.device("cuda"))
    answerer_1 = guesser_1
    guesser_2 = guesser_1
    answerer_2 = answerer_1
    env.run([guesser_1, answerer_1, guesser_2, answerer_2])
    print("----------------------------------------")
    print(env.render(mode="ansi"))


def main():
    args = create_argparser().parse_args()
    play(debug=args.debug)


if __name__ == "__main__":
    main()
