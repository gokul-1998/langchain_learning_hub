from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# 1st prompt
template1=PromptTemplate(template="Write a detailed report on {topic}",
                        input_variables=['topic']
                        )


template2=PromptTemplate(template="write a 4 point summary on the following {text}",
                        input_variables=["text"]
                        )

parser=StrOutputParser()

# chain
chain=template1 | model | parser | template2 | model | parser


result=chain.invoke({'topic':"English Premier League 2023/2024"})

print(result)