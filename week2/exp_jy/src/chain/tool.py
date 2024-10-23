import os
from dotenv import load_dotenv
import yaml
import pandas as pd
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from chain.model import llm
from chain.output_parser import JSONParser


load_dotenv()


# with open("configs/database.yaml", "r") as f:
#     config = yaml.safe_load(f)

# if not os.path.exists(config["URI_PATH"]):
#     with sqlite3.connect(config["URI_PATH"]) as conn:
#         df = pd.read_csv(config["CSV_PATH"])[
#             [
#                 "title",
#                 "director",
#                 "screenwriter",
#                 "plot",
#                 "rating",
#                 "rating_count",
#                 "actors",
#                 "genres",
#                 "countries",
#                 "audience",
#                 "running_time",
#                 "adult",
#             ]
#         ]
#         df.to_sql("movie", conn, if_exists="replace", index=False)

# movie_db = SQLDatabase.from_uri(f"sqlite:///{config['URI_PATH']}")
# movie_db_tool = QuerySQLDataBaseTool(db=movie_db)

metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the movie.",
        type="string",
    ),
    AttributeInfo(
        name="director",
        description="A list of the movie's directors.",
        type="list",
    ),
    AttributeInfo(
        name="screenwriter",
        description="A list of the movie's screenwriters.",
        type="list",
    ),
    AttributeInfo(
        name="plot",
        description="A short summary of the movie's plot.",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="The average rating given to the movie. Range 0.0 ~ 5.0",
        type="float",
    ),
    AttributeInfo(
        name="rating_count",
        description="The total number of ratings the movie has received.",
        type="float",
    ),
    AttributeInfo(
        name="actors",
        description="A list of the movie's main actors.",
        type="list",
    ),
    AttributeInfo(
        name="genres",
        description="A list of the genres.",
        type="list",
    ),
    AttributeInfo(
        name="countries",
        description="A list of the countries where the movie was produced.",
        type="list",
    ),
    AttributeInfo(
        name="audience",
        description="Cumulative audience",
        type="float",
    ),
    AttributeInfo(
        name="running_time",
        description="The running time of the movie in minutes.",
        type="float",
    ),
    AttributeInfo(
        name="adult",
        description="A flag indicating whether the movie is for adults only.",
        type="float",
    ),
]
content_description = "plot and reviews of the movie"

from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings

contexts_df = pd.read_csv("./data/movies_sample_cleaned_chroma.csv")
loader = DataFrameLoader(contexts_df, page_content_column="plot_review")

vectorstore = Chroma.from_documents(
    loader.load(),
    # HuggingFaceEmbeddings(
    #     model_name="upskyy/bge-m3-korean", model_kwargs={"device": "cuda"}
    # ),
    OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./data/chroma",
)


movie_retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, content_description, metadata_field_info
)


# from langchain.chains.query_constructor.base import (
#     StructuredQueryOutputParser,
#     get_query_constructor_prompt,
# )

# prompt = get_query_constructor_prompt(
#     content_description,
#     metadata_field_info,
# )
# output_parser = StructuredQueryOutputParser.from_components()
# query_constructor = prompt | llm | output_parser

# from langchain_community.query_constructors.chroma import ChromaTranslator

# movie_retriever = SelfQueryRetriever(
#     query_constructor=query_constructor,
#     vectorstore=vectorstore,
#     structured_query_translator=ChromaTranslator(),
# )
