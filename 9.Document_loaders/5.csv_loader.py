from langchain_community.document_loaders import CSVLoader

loader=CSVLoader("data.csv")

data=loader.load()

print(data)