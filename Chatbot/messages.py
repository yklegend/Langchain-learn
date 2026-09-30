from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2
)
model = ChatHuggingFace(llm = llm)

messages =[
    SystemMessage(content='You are a helpful doctor'),
    HumanMessage(content = 'Tell me about different communicable diseases')

]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))
print(messages)
