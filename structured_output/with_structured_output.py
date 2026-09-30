from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2
)
model = ChatHuggingFace(llm = llm)

#Schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str,"A brief summary of the review"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside the list"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either positive or negative"]
structured_model = model.with_structured_output(Review)





    
result = model.invoke("""The hardware is great but the software feels bloated. There are too many pre-installed apps that I can't remove.""")
print(result)
