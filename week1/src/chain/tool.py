import os
from dotenv import load_dotenv
import yaml
import pandas as pd
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()


with open("configs/database.yaml", "r") as f:
    config = yaml.safe_load(f)

if not os.path.exists(config["URI_PATH"]):
    with sqlite3.connect(config["URI_PATH"]) as conn:
        df = pd.read_csv(config["CSV_PATH"])[
            [
                "title",
                "director",
                "screenwriter",
                "plot",
                "rating",
                "rating_count",
                "actors",
                "genres",
                "countries",
                "audience",
                "running_time",
                "adult",
            ]
        ]
        df.to_sql("movie", conn, if_exists="replace", index=False)

movie_db = SQLDatabase.from_uri(f"sqlite:///{config['URI_PATH']}")
movie_db_tool = QuerySQLDataBaseTool(db=movie_db)


embeddings = HuggingFaceEmbeddings(model="BAAI/bge-m3")
