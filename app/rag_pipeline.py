from retrieval import retrieve_documents, encode_documents, create_index
from generation import generate_answer
from dataset import load_dataset

def rag_pipeline(query):
    dataset = load_dataset()
    embeddings = encode_documents(dataset)
    index = create_index(embeddings)
    
    retrieved_docs = retrieve_documents(query, index, dataset)
    answer = generate_answer(query, retrieved_docs)
    
    return answer
