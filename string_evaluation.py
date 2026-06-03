from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM
eval_llm = ChatOllama(model="gemma3:4b", base_url="http://localhost:11434")

# Prompt
prompt_template = PromptTemplate.from_template("""
You are evaluating whether the predicted answer is correct given the reference answer.

Question: {input}
Reference Answer: {reference}
Predicted Answer: {prediction}

Is the prediction correct? Answer with CORRECT or INCORRECT and a brief reason.
""")

# Chain
eval_chain = prompt_template | eval_llm | StrOutputParser()

query = "How Iran Controls a major part of Oil Export in the World?"

predicted_answer = "Iran controls a major part of oil exports through its vast oil reserves, membership in OPEC, and state-owned National Iranian Oil Company (NIOC) which manages production and exports and Controling ships in Strit of Hormuz via his Powerfull army, missiles, and drones."

ref_answer = "Iran holds one of the world's largest oil reserves and uses OPEC membership and its state oil company to influence global oil export markets and also controls hormuz via his Powerfull army, missiles, and drones"

result = eval_chain.invoke({
    "input": query,
    "prediction": predicted_answer,
    "reference": ref_answer
})

print(f"Score: {result}")