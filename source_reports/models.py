"""Native reasoning and final-answer boundaries for the two model families."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    model_id: str
    revision: str
    dtype: str
    top_p: float
    repetition_penalty: float
    language_model_only: bool = False


PROFILES = {
    "gemma": Profile(
        "gemma", "google/gemma-4-E2B-it",
        "70af34e20bd4b7a91f0de6b22675850c43922a03",
        "bfloat16", 1.0, 1.0, True,
    ),
    "gpt-oss": Profile(
        "gpt-oss", "openai/gpt-oss-20b",
        "6cee5e81ee83917806bbde320786a8fb61efebee",
        "auto", 0.95, 1.05,
    ),
}


def find_tokens(haystack: list[int], needle: list[int], start: int = 0) -> tuple[int, int]:
    for index in range(start, len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == needle:
            return index, index + len(needle)
    raise ValueError("Expected reasoning channel marker was not found")


class NativeReasoning:
    def __init__(self, profile: Profile):
        from transformers import AutoProcessor, AutoTokenizer

        self.profile = profile
        if profile.name == "gemma":
            self.processor = AutoProcessor.from_pretrained(
                profile.model_id, revision=profile.revision
            )
            self.tokenizer = self.processor.tokenizer
            self.opening = self._encode("<|channel>thought\n")
            self.boundary = [self.tokenizer.convert_tokens_to_ids("<channel|>")]
            self.stop_ids = [
                self.tokenizer.convert_tokens_to_ids("<turn|>"),
                self.tokenizer.eos_token_id,
            ]
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                profile.model_id, revision=profile.revision
            )
            self.opening = self._encode("<|channel|>analysis<|message|>")
            self.boundary = self._encode(
                "<|end|><|start|>assistant<|channel|>final<|message|>"
            )
            self.stop_ids = [self.tokenizer.eos_token_id]
        if not self.opening or not self.boundary or any(v is None for v in self.stop_ids):
            raise ValueError("Incomplete reasoning channel configuration")

    def _encode(self, value: str) -> list[int]:
        return list(self.tokenizer.encode(value, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def question_prompt(self, question: str) -> list[int]:
        if self.profile.name == "gemma":
            encoded = self.processor.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_tensors="pt",
            )
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            prompt = list(encoded[0].tolist())
        else:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="medium",
            )
            prompt = self._encode(rendered)
        if prompt[-len(self.opening):] == self.opening:
            raise ValueError("The chat template already opened a reasoning channel")
        return prompt

    def source_parts(self, output_ids: list[int]) -> tuple[list[int], list[int]]:
        _, start = find_tokens(output_ids, self.opening)
        end, answer_start = find_tokens(output_ids, self.boundary, start)
        if end == start:
            raise ValueError("The source response contains an empty reasoning trace")
        if answer_start == len(output_ids):
            raise ValueError("The source response contains an empty final answer")
        return output_ids[start:end], output_ids[answer_start:]

    def report_prompt(self, question: str, trace_ids: list[int], instruction: str) -> list[int]:
        question_ids = self.question_prompt(question + "\n\n" + instruction)
        return question_ids + self.opening + trace_ids + self.boundary
