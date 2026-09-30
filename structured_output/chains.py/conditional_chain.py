from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description ='Give the sentimnet of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n{format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2
#result = classifier_chain.invoke({'feedback': 'This is a great smartphone'}).sentiment
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)


prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment =='Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a very good phone'}))