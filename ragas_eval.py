from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


llm = LangchainLLMWrapper(Ollama(model="gemma3:4b", base_url="http://localhost:11434"))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="gemma3:4b", base_url="http://localhost:11434"))

data = {
    "question": ["How Iran Controls Oil Export in the World?"],
    "answer":   ["Iran Controls Oil Export in the World by Restricting Strait of Hormuz"],
    "contexts": [[
        "Iran Controls Oil Export in the World by Closing Strait of Hormuz whenever he Wants",
        "Iran Deploy his Army power in the Strait of Hormuz to monitor it,"
    ]],
    "ground_truth": ["Iran Controls Oil Export in the World by Restricting Strait of Hormuz via Deploy his Army power in the zone"]
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall, faithfulness],
    llm=llm,
    embeddings=embeddings,
)

print(result)