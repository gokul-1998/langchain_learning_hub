from langchain_classic import text_splitter
from google.genai import documents
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA    

# step 1 : load the credentials
load_dotenv()

# step 2: Load the document
loader=TextLoader('docs.txt')
documents=loader.load()


# step 3 : Split the text into smaller chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_splitter.split_documents(documents)

# step 4 : Convert text embeddings and store in FAISS
embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore=FAISS.from_documents(docs,embeddings)

# step 5: create a retriever (fetches relevant documents)
retriever=vectorstore.as_retriever()

# step 6: Initialize a model
model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# step 7 : Create a Retrieval QA chain

chain=RetrievalQA.from_chain_type(llm=model,retriever=retriever)

# step 8: Manually query the model and retrieve relevant documents
# query= "what are the key takeaways from the documents?"
query="what kind of technologies does AI comprise of?"
response=chain.invoke(query)

# step 9 : print answer

print(response)