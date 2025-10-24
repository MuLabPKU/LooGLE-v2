import time
import requests
from openai import OpenAI


class LLMClient:

    def __init__(self, model, base_url, api_key):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def query(self, prompt, temperature=0.1, max_tokens=32, max_retries=5):
        for attempt in range(max_retries):
            try:
                return self._call_api(prompt, temperature, max_tokens)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error: {str(e)} - Retry {attempt + 1}/{max_retries}")
                time.sleep(1)

        print("Max retries reached. Failed.")
        return ''

    def _call_api(self, prompt, temperature, max_tokens):
        model = self.model

        if "Kimi" in model:
            return self._call_kimi(prompt, temperature, max_tokens)

        if any(x in model for x in ["o1", "o3", "o4"]):
            response = self.client.responses.create(model=model, input=prompt)
            print(response.usage)
            return response.output_text

        if "gpt" in model:
            response = self.client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature
            )
            print(response.usage)
            return response.output_text

        if "Yarn-Mistral" in model or "Yi" in model:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text

        if model == "deepseek-ai/DeepSeek-V3":
            model = "deepseek-chat"
        elif model == "deepseek-ai/DeepSeek-R1":
            model = "deepseek-reasoner"

        completion = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return completion.choices[0].message.content

    def _call_kimi(self, prompt, temperature, max_tokens):
        url = str(self.client.base_url) + "chat/completions/"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]
