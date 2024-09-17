import faiss
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

def encode_documents(dataset):
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    encoded_docs = []
    for doc in dataset['text']:
        inputs = ctx_tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
        embedding = ctx_encoder(**inputs).pooler_output
        encoded_docs.append(embedding.detach().numpy())
    
    return np.concatenate(encoded_docs, axis=0)

def create_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_documents(query, index, dataset, top_k=2):
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    inputs = ctx_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_embedding = ctx_encoder(**inputs).pooler_output.detach().numpy()

    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = [dataset['text'][i] for i in indices[0]]
    return retrieved_docs
