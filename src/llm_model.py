from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        context = "\n---\n".join([chunk['content'] for chunk in chunks])
        return (f"""
            CONTEXT:
            {context}

            QUESTION: {query}
            ANSWER:
        """)

    def generate(self, query: str,
                 chunks: list[dict], max_tokens: int = 256) -> str:
        prompt = self._build_prompt(query, chunks)
        messages = [
            {
                "role": "system",
                "content": "You are an expert programming assistant."
                "Answer the user's question using ONLY the provided"
                "context. Do not use outside knowledge."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        prompt_enhanced = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(
            prompt_enhanced, return_tensors="pt"
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        tokens = outputs[0][inputs.input_ids.shape[-1]:]

        return str(self.tokenizer.decode(tokens, skip_special_tokens=True))
