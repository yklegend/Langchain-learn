from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate 
# In latest versions, it is often found here:
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=1.2
)
model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name='fact_1', description= 'Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description= 'Fact 2 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 2 facts about {topic} \n {format_instruction}',
    input_variables= ['topic'],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}

)

chain = template | model | parser

result = chain.invoke({'topic' : 'sun'})
print(result)

