import zhipuai
import openai


class HypnosisAttacker:
    def __init__(self, model: str, key: str) -> None:
        self.model = "ChatGPT" if "gpt" in model else "Zhipu AI"
        if "gpt" in model:
            self.client = openai.OpenAI(api_key=key)
        else:
            self.client = zhipuai.ZhipuAI(api_key=key)

    def run(self, prompt: str, attack_model: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=attack_model,
            messages=[
                {"role": "system", "content": system_prompt.format(model=self.model)},
                {"role": "user", "content": prompt},
            ],
        )
        return {"prompt": prompt, "response": response.choices[0].message.content}
