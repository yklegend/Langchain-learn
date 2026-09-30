from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    temperature=1.2
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Generate 5 facts about {Topic}",
    input_variables=['Topic']
)

chain = prompt | model | parser

result = chain.invoke({'Topic' : 'germany'})

print(result)

chain.get_graph().print_ascii()