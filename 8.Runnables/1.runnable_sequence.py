from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence

load_dotenv()

prompt=PromptTemplate(template="write a joke about a {topic}",input_variables=['topic'])

prompt2=PromptTemplate(template="Explain the following joke \n {joke}",input_variables=['joke'])

model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")

parser=StrOutputParser()

chain=RunnableSequence(prompt,model,parser,prompt2,model,parser)
# chain = prompt | model | parser | prompt2 | model | parser


result=chain.invoke({"topic":"monkey"})
print(result)