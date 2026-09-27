from urllib import response
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")

prompt=PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=['topic']
)


chain=LLMChain(
    llm=model,prompt=prompt
)

topic="31 Atlas Interstellar Object"

response=chain.invoke({"topic":topic})
print(response)