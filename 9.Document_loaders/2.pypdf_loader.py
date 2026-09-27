from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

parser=StrOutputParser()
loader=PyPDFLoader('Agentic AI Problem Statement.pdf')

docs=loader.load()

print(docs)