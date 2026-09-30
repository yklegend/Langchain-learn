from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence



load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template='Write a joke about {Topic}',
    input_variables = ['Topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the joke - {Text}',
    input_variables=['Text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)
print(chain.invoke({'Topic': 'SUN'}))

