from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text="Delhi is the capital of India"

documents=[
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
    "I love pizza"
]

result=embeddings.embed_query(text)

# print(str(result))

result_doc=embeddings.embed_documents(documents)


similarity_scores=cosine_similarity([result],result_doc)

print(similarity_scores)