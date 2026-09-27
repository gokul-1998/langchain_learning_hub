from langchain_community.retrievers import WikipediaRetriever

r=WikipediaRetriever(top_k_results=2,lang="en")

query="Indian Premier League"

docs=r.invoke(query)

print(len(docs))