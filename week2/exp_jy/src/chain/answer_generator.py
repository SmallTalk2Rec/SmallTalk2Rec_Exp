from langchain_core.prompts import ChatPromptTemplate

from chain.model import llm
from chain.output_parser import JSONParser

SYSTEM_TEMPLATE = """You are an AI assistant specializing in movie information. Your task is to respond to the user's questions in friendly manner."""

USER_TEMPLATE = """<Instruction>
Please recommend movie from query and context. format your response using markdown so that it is easy for users to read and understand. Please respond only in the given JSON response format.
</Instruction>

<Context>
{context}
</Context>

<Query>
{question}
</Query>

<JSON_Response_Format>
{{
    "answer": answer
}}
</JSON_Response_Format>
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("user", USER_TEMPLATE),
    ]
)

generate_answer_chain = prompt | llm | JSONParser()
