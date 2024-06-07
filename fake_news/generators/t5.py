from transformers import (  # type: ignore
    T5Tokenizer,
    T5ForConditionalGeneration,
    GenerationConfig,
)

from typing import Any

from fake_news.base import AbstractNewsGenerator


class T5Generator(AbstractNewsGenerator):
    def __init__(
        self,
        model_path: str = "AmevinLS/flan-t5-base-realnews",
        device_map: str = "auto",
    ):
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map=device_map
        )
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_path)

    def generate(
        self,
        titles: list[str],
        generation_config: GenerationConfig | dict[str, Any] | None = None,
    ) -> list[str]:
        if isinstance(titles, str):
            titles = [titles]
        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        elif generation_config is None:
            generation_config = GenerationConfig()

        input_tokens = self.tokenizer(
            [self._format_prompt(title) for title in titles],
            return_tensors="pt",
            padding=True,
        )["input_ids"]
        input_tokens = input_tokens.to(device=self.model.device)
        output_tokens = self.model.generate(
            input_tokens, generation_config=generation_config
        )

        return self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )

    def _format_prompt(self, title: str):
        PREFIX = "Please write an article based on the title: "
        return PREFIX + title


if __name__ == "__main__":
    generator = T5Generator()
    output = generator.generate(
        ["Belarus under new set of sanctions"],
        {
            "max_new_tokens": 200,
            "repetition_penalty": 2.0,
        },
    )
    print(output)
