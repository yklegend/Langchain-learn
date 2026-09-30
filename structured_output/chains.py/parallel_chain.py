from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)
model1 = ChatHuggingFace(llm = llm1)
model2 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template='Generate short notes on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']

)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'topic': 'Switzerland'})
print(result)

