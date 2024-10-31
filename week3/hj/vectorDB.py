"""
vectorDB 저장
"""

import pandas as pd

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings

import os


## 줄거리 embedding 생성
model_name = 'jhgan/ko-sroberta-multitask'
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}
model_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

df = pd.read_csv('../../../files/preprocessed_movie_sample.csv')
loader = DataFrameLoader(df, page_content_column="plot_review")

database = Chroma.from_documents(
    loader.load(),
    persist_directory="../../../files/chroma",
)