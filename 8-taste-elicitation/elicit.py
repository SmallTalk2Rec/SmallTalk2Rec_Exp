import os
from dotenv import load_dotenv
import re
import json

import openai

from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

import chainlit as cl



## Load API
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

manager = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature = 0
)

reply_bot = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature = 0
)


## 0) subtask manager

# 기본적인 소통을 위한 페르소나 설정도 들어가있으면 좋을 듯
hj_subtask = '''The AI assistant should analyze the dialogue context to detect which sub-task should be selected.

Sub-task name: It provides a unique identifier for sub-task detection, which is used for references to related expert models and their predicted results.
Thess are the subtask lists. [ User preference elicitation, Recommendation, Explanation, Item detail search ]

There are some cases for your reference.

user_query: I want to find a legal drama
answer: user preference elicitation

user_query: I want the female actor as the lead role
answer: user preference elicitation

user_query: I can't wait anymore. Give me reccomendations
answer: recommendation

user_query: Why do you recommend it to me?
answer: explanation

user_query: OK. what timeframe is it?
answer: item detail research

Which sub-task should be selected? The sub-task MUST be selected from the above list.
user_query: {query}
answer: 
'''

def subtask_detect(query):
    '''
    subtask detection module
    input
        query: 유저 입력문
    '''

    subtask_detection_prompt = PromptTemplate(
        template  = hj_subtask,
        input_variables = ['query']
    )

    # Chain 정의
    chain = LLMChain(
        llm = manager,
        prompt = subtask_detection_prompt
    )

    result = chain.invoke({'query':query}) # chain 실행
    return result['text']



## 1) taste elicitation

mk_scoring = """
You are a movie preference evaluator. When a user provides a preference for a specific movie, you will assess the given movie and return a JSON object that details the user's preference evaluation using English.

The answer should include the following fields:

content: Categorize the content among [director, actor, genre, movie_title]
item: A name of the content
preference_score: A numerical score indicating how much the user prefers the movie on a scale of 1 to 10.
comments: A brief explanation or comments about why the user has this preference score for the movie.

user input: 어제 에일리언을 봤는데 꽤 재밌더라구. 엄청 흥미진진해서 시간가는 줄 몰랐어
answer: 
 'content': 'movie_title'
 'item': 'Alien',
 'preference_score': 8,
 'comments': 'The user found 'Alien' to be quite entertaining and thrilling, indicating a strong engagement with the film's suspenseful elements.'

user input: 내가 가장 좋아하는 장르는 로맨스야. 절절한 사랑 이야기가 너무 좋아
answer: 
 'content':'genre',
 'item': 'romance',
 'preference_score': 9,
 'comments': 'The user loves heartfelt love stories which beautifully captures deep emotional connections and enduring love.'

user input: "신과함께 같은 영화는 너무 싫어.... 신파가 너무 많아")
 'content':'movie_title',
 'item': '신과함께',
 'preference_score': 2,
 'comments': 'The user dislikes this movie due to its excessive melodrama, which they find overwhelming.'

user input: {query}
answer: 
"""

def taste_elicit(query):

    taste_elicitation = PromptTemplate(
        template  = mk_scoring,
        input_variables = ['query']
    )

    # Chain 정의
    subtask_chain = LLMChain(
        llm = manager,
        prompt = taste_elicitation
    )

    result = subtask_chain.invoke({'query': query}) # chain 실행

    pattern = r'^```json\s*|\s*```$'
    cleaned_text = re.sub(pattern, '', result['text'], flags=re.DOTALL).strip()
    parsed_data = json.loads(cleaned_text)

    return parsed_data

user_profile = {
    'movie_title': [],
    'director': [],
    'actor': [],
    'genre': [],
    'requirements': []
}


## 2) Reply Agent

reply_message = """This is an movie recommendation task and your assignment is to interact and ask user's taste. Follow several rules below.

1) Treat user as if you are an counselor.
2) AI assitant should ask questions to fill the user profile dictionary.
3) There are 4 necessary items and 1 optional item.
    neccesary items: [movie_title, director, actor, genre]
    optional item: requirements
4) Speak Korean

[user profile]
movie_title: {movie_title}
director: {director}
actor: {actor}
genre: {genre}

your question:
"""

def reply(user_profile):
    '''
    reply module
    input
        user_profile: 현재까지 elicit한 결과물
    '''
    reply_prompt = PromptTemplate(
        template  = reply_message,
        input_variables = ['movie_title','director','actor','genre']
    )

    reply_chain = LLMChain(
        llm = reply_bot,
        prompt = reply_prompt
    )

    result = reply_chain.invoke({'movie_title': user_profile['movie_title'], 'director': user_profile['director'], 'actor': user_profile['actor'], 'genre': user_profile['genre']})
    return result['text']

## 3) Chainlit
Q = {}
Q['start'] = 'Welcome to SmallTalk Lab! \nPlease let me know about your "movie tastes"!'
Q['movie_title'] = 'What is your favorite movie?'
Q['director'] = 'Who is your favorite director?'
Q['actor'] = 'Who is your favorite actors/actresses'
Q['genre'] = 'What is your favorite genre?'
Q['requirements'] = 'Give me any requirements'



@cl.on_chat_start
async def on_chat_start():    
    await cl.Message(content = Q['start']).send()


@cl.on_message
async def on_message(message):

    utterance = message.content
    subtask_type = subtask_detect(utterance)
    print(subtask_type)

    # 이 부분 훨씬 고급지게 callback 문으로 처리할 수 있을 것 같은데
    if 'user preference elicitation' in subtask_type.lower():
        taste_temp = taste_elicit(utterance)
        print(taste_temp)
        user_profile[taste_temp['content']].append((taste_temp['item'], taste_temp['preference_score']))
    
    # Reply Agent 대체 반복문
    # for subject in user_profile:
    #     if not user_profile[subject]:
    #         await cl.Message(content = Q[subject]).send()
    #         break
    

    # Reply Agent
    reply2user = reply(user_profile)
    print(user_profile)
    await cl.Message(content = reply2user).send()