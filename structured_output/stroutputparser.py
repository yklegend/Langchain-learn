from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 

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

prompt1 = template1.invoke({'topic' : 'Blackhole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result.content})

result1 = model.invoke(prompt2)

print(result1.content)
