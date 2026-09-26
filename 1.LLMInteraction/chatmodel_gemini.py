from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt="My name is Abhishek, what is your name?"

result=model.invoke(prompt)
print(result)
