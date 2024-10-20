from chain.sql_query_generator import generate_sql_query_chain
from chain.answer_generator import generate_answer_chain
from chain.tool import movie_db_tool
from graph.state import GraphState


def get_movie(state: GraphState) -> GraphState:
    response = generate_sql_query_chain.invoke(
        {
            "question": state["question"],
        }
    )
    context = movie_db_tool.invoke({"query": response})
    return GraphState(sql_query=response, context=context)


def get_answer(state: GraphState) -> GraphState:
    response = generate_answer_chain.invoke(
        {
            "question": state["question"],
            "sql_query": state["sql_query"],
            "context": state["context"],
        }
    )
    return GraphState(answer=response["answer"])
