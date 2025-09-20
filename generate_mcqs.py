import re
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

MCQ_PROMPT = """
You are an educational assistant. Based on the following context from an NCERT textbook, generate a **single multiple-choice question** related to **sustainable development**.

Requirements:
- Make sure the question is factual and based only on the context.
- Provide 4 answer options labeled A, B, C, and D.
- Clearly mark the correct answer.
- Keep the language suitable for school students.

Context:
{context}

---
Return in this format:

Question: ...
A) ...
B) ...
C) ...
D) ...
Answer: ...
"""

def generate_mcq(topic: str = "sustainable development"):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Retrieve relevant chunks
    results = db.similarity_search_with_score(topic, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(MCQ_PROMPT)
    prompt = prompt_template.format(context=context_text)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    mcq_data = parse_mcq(response_text)
    return mcq_data
    # return response_text

# print(generate_mcq())

def parse_mcq(response_text: str):
    # Basic regex to parse options and answer
    try:
        question = re.search(r"Question:\s*(.*)", response_text).group(1).strip()
        options = {
            "A": re.search(r"A\)\s*(.*)", response_text).group(1).strip(),
            "B": re.search(r"B\)\s*(.*)", response_text).group(1).strip(),
            "C": re.search(r"C\)\s*(.*)", response_text).group(1).strip(),
            "D": re.search(r"D\)\s*(.*)", response_text).group(1).strip(),
        }
        answer = re.search(r"Answer:\s*([ABCD])", response_text).group(1).strip()
        return {
            "question": question,
            "options": options,
            "correct": answer
        }
    except Exception as e:
        print("Failed to parse MCQ:", e)
        return None