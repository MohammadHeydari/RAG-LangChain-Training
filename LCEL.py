import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

class OllamaRunnable(Runnable):
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def invoke(self, input, config=None):

        url = f"{self.base_url}/api/generate"

        # 🔥 تبدیل به string
        prompt_str = input.to_string()

        payload = {
            "model": self.model,
            "prompt": prompt_str,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 200,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        return data.get("response", "")


prompt = """
Use only the context provided to answer the following question.
If you don't know the answer, reply that you are unsure.

Context: {context}
Question: {question}
"""

prompt_template = ChatPromptTemplate.from_template(prompt)

llm = OllamaRunnable(model="gemma3:4b")

chain = prompt_template | llm

result = chain.invoke({
    "context": "Mohammad Heydari is an AI Data Engineer at SiCS Company",
    "question": "Where does Mohammad work?"
})

print("Result:")
print(result)