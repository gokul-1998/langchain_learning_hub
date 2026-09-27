import wikipedia
from langchain_community.retrievers import WikipediaRetriever

# Wikimedia blocks the default user-agent with HTTP 429, causing JSONDecodeError
wikipedia.set_user_agent("LangChainLearningHub/1.0 (contact: user@example.com)")

r=WikipediaRetriever(top_k_results=2,lang="en")

query="Indian Premier League"

docs=r.invoke(query)

# print(len(docs))

for i,doc in enumerate(docs):
    print(f"--------- result :{i+1}")
    print("content",doc.page_content)