from langchain_core.prompts import ChatPromptTemplate

from chain.model import llm
from chain.output_parser import JSONParser

SYSTEM_TEMPLATE = """You are an AI assistant specializing in movie information. Your task is to respond to the user's questions in friendly manner."""

USER_TEMPLATE = """<Instruction>
Please Answer the Question based on the SQL_Query and the Data_obtained_through_the_query from "movie" Table. format your response using markdown so that it is easy for users to read and understand. Please respond only in the given JSON response format.
</Instruction>

<Description_of_Each_Column_in_the_Movie_Table>
title: The title of the movie. (TEXT)
director: A list of the movie's directors. (LIST)
screenwriter: A list of the movie's screenwriters. (LIST)
plot: A short summary of the movie's plot. (TEXT)
rating: The average rating given to the movie. Range 0.0 ~ 5.0 (FLOAT)
rating_count: The total number of ratings the movie has received. (FLOAT)
actors: A list of the movie's main actors. (LIST)
genres: A list of the genres. (LIST)
countries: A list of the countries where the movie was produced. (LIST)
audience: Cumulative audience (FLOAT)
running_time: The running time of the movie in minutes. (FLOAT)
adult: A flag indicating whether the movie is for adults only (1.0 for true, 0.0 for false). (FLOAT)
</Description_of_Each_Column_in_the_Movie_Table>

<SQL_Query>
{sql_query}
</SQL_Query>

<Data_obtained_through_the_query>
{context}
</Data_obtained_through_the_query>

<Question>
{question}
</Question>

<JSON_Response_Format>
{{
    "answer": "Your answer here"
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
