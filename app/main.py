from rag_pipeline import rag_pipeline

if __name__ == "__main__":
    query = "Aprendizado"
    response = rag_pipeline(query)
    print(f"Resposta: {response}")
