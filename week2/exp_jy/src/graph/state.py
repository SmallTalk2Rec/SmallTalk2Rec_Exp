from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
    """

    question: str
    answer: str
    sql_query: str
    context: str
