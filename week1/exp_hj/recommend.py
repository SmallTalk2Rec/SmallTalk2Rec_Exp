import duckdb
import openai
from langchain_openai import ChatOpenAI
from langchain import LLMChain, PromptTemplate

from dotenv import load_dotenv
import os

import chainlit as cl

'''
명령어: chainlit run recommend.py
'''

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

# 0) 초기설정
@cl.on_chat_start
async def on_chat_start():    
    await cl.Message(content = '어떤 영화를 추천해드릴까요?').send()
    
@cl.on_message
async def on_message(message):

    user_query = message.content
    keywords = user_tagging(user_query).split('||')
    movies = recommend(keywords)
    await cl.Message(content= movies).send()


db_file_path = 'week1.duckdb'
con = duckdb.connect(db_file_path)


# 1) 요청을 해시태그로 변형하기
def user_tagging(user_query):
    user_tagger = ChatOpenAI(model= 'gpt-4o-mini', temperature= 0)

    usertag_prompt = """영화 추천할거야. 유저의 요청을 감독, 각본가, 장르, 배우, 감성 관련해서 3개이하의 키워드로 변형해줘. 각 키워드는 ||로 구분해줘

    user: 레이첼 맥아담스 나오는 영화 추천해줘
    answer: 레이첼 맥아담스

    user: 신나는 블록버스터 영화 추천해줘
    answer: 블록버스터

    user: '노트북'이랑 비슷한 영화 추천해줘
    answer: 로맨스||레이첼 맥아담스||라이언 고슬링

    user: 스타워즈 나 너무 좋아하는데. 비슷한 영화 추천해줘.
    answer: SF||판타지

    user: 탑건이랑 비슷한 영화 추천해줘.
    answer: 톰 크루즈||액션||전투기

    user: {query}
    answer: """

    usertag_template = PromptTemplate(
        template  = usertag_prompt,
        input_variables = ['query']
    )

    query_making = LLMChain(
        llm = user_tagger,
        prompt = usertag_template
    )

    result = query_making.invoke(user_query)
    return result['text']



# 2) SQL 쿼리로 추천하기
def recommend(keywords):
    recommends = []

    for keyword in keywords:
        query = f"SELECT Title FROM movies WHERE Overall LIKE '%{keyword}%'"
        recommends.append(con.execute(query).fetchall())

    # 연결 종료
    con.close()
    return recommends