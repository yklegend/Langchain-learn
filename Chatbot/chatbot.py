from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2,
    max_new_tokens= 10
)
model = ChatHuggingFace(llm = llm)

chat_history =[
    SystemMessage(content = 'You are helpful chatbot')
]

while True:
    user_input = input("You:")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content = result.content))
    print("AI:", result.content)


print(chat_history)
