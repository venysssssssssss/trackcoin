from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_answer(query, retrieved_docs):
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    context = " ".join(retrieved_docs)
    input_text = f"question: {query} context: {context}"
    
    inputs = t5_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    output_ids = t5_model.generate(inputs["input_ids"], max_length=50)
    
    answer = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer
