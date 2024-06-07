from transformers import (  # type: ignore
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
)

from typing import Any

from fake_news.base import AbstractNewsGenerator


class TinyLlamaGenerator(AbstractNewsGenerator):
    def __init__(
        self,
        model_path: str = "AmevinLS/TinyLLama-lora-realnews",
        device_map: str = "auto",
    ):
        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
            model_path, device_map=device_map
        )  # NOTE: you must have 'peft' installed to load adapter models
        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
            model_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._prefix = "### Title:"
        self._response_template = "\n### Article: "

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

        model_outputs = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )

        return [
            self._extract_article(model_output)
            for model_output in model_outputs
        ]

    def _format_prompt(self, title: str):
        return f"{self._prefix} {title}. {self._response_template}"

    def _extract_article(self, model_output: str):
        return model_output.split(self._response_template)[-1]


if __name__ == "__main__":
    # import torch
    # print(torch.cuda.is_available())
    generator = TinyLlamaGenerator()
    output = generator.generate(
        ["Belarus under new set of sanctions"],
        {
            "max_new_tokens": 200,
            "repetition_penalty": 2.0,
        },
    )
    print(output)
