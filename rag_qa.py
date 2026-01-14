from ingest import load_pdf, chunk_text
from embed_store import VectorStore
from transformers import pipeline
import torch

PDF_PATH = "data/Backend_Roadmap.pdf"

def main():
    print("📄 Loading PDF...")
    text = load_pdf(PDF_PATH)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("🧠 Building vector store...")
    store = VectorStore()
    store.build(chunks)

    print("🤖 Loading LLM...")
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if torch.cuda.is_available() else -1
    )

    print("\n✅ RAG SYSTEM READY")
    print("Type a question (or 'exit')\n")

    while True:
        question = input(">> ")
        if question.lower() == "exit":
            break

        context_chunks = store.retrieve(question)

        if not context_chunks:
            print("\nANSWER:\nI don't know. The document does not contain this information.\n")
            continue

        context = "\n\n".join(context_chunks)

        prompt = f"""
Context:
{context}

Question:
{question}

Answer clearly in 2–3 sentences:
"""

        result = llm(prompt, max_new_tokens=150)
        print("\nANSWER:\n", result[0]["generated_text"], "\n")


if __name__ == "__main__":
    main()
