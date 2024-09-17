from datasets import Dataset

def load_dataset():
    documents = [
        {"text": "Aprendizado de máquina é uma subárea da inteligência artificial."},
        {"text": "A Terra é o terceiro planeta mais próximo do Sol."},
        {"text": "Python é uma linguagem de programação usada para diversas finalidades."},
    ]
    return Dataset.from_dict({"text": [doc["text"] for doc in documents]})
