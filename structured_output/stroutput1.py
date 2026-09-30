from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2
)
model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template= 'Write a detailed report on th topic {topic}',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'Write a 5 line summary on the following text. /n {text}',
    input_variables= ['text']
)
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'moon'})

print(result)