from langchain_core.prompts import ChatPromptTemplate

from chain.model import llm
from chain.output_parser import SqlParser

SYSTEM_TEMPLATE = """You are a SQLite expert. Your task is to create a syntactically correct SQLite query to run."""

USER_TEMPLATE = """<Instruction>
Please create a SQLite query to retrieve the necessary information from the "movie" table to answer the question. Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per SQLite.
</Instruction>

<Description_of_the_Movie_Table>
The "movie" table contains data on various movies, including metadata such as director, screenwriter, plot, and ratings.
</Description_of_the_Movie_Table>

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

<Question>
{question}
</Question>

<Response_Format>
SQL Query to run
</Response_Format>
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("user", USER_TEMPLATE),
    ]
)


generate_sql_query_chain = prompt | llm | SqlParser()
