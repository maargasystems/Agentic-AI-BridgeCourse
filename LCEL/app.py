from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes



load_dotenv(override=True)

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

model = ChatGroq(model="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_messages([("system", "act as a {language} language class teacher and translate the text please?"),
                                           ("user", "please provide the english text {text}")])

parser = StrOutputParser()

chain = prompt|model|parser

app = FastAPI(title="langchain server", version="1.0", description="A simple API server using Langchain runnable interfaces")

add_routes(app, chain, path="/chain")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)

