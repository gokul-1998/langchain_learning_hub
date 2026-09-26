from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# 1st prompt
template1=PromptTemplate(template="Write a detailed report on {topic}",
                        input_variables=['topic']
                        )
prompt1=template1.invoke({"topic":"English Premier League 2023/2024"})

result1=model.invoke(prompt1).content

print(result1)
print("*"*50)

# 2nd prompt

template2=PromptTemplate(template="write a 4 point summary on the following {text}",
                        input_variables=["text"]
                        )

prompt2=template2.invoke({"text":str(result1)})

result=model.invoke(prompt2)

print(result.content)