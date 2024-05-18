# Graph state

from typing import List
from PK_LLM_endfront.loggs import logger

from PK_LLM_endfront.Langgraph.Nodes import Intent_analyze,Router,Retrievers,Reranker,\
    _Web_search,Web_searcher,Generater_dirctly,Generater
from PK_LLM_endfront.Langgraph.Edges import Retrieval_manager,Web_search_manager
from typing_extensions import TypedDict
from typing import List

class Graph_State(TypedDict):
    """图状态
    Args:
        question: question
        history
        context
    """
    retrieval:bool
    question:str
    history:str
    context:List[str] =[]
    output:str
    domain:List[int]


# 初始化
from langgraph.graph import END, StateGraph
workflow = StateGraph(Graph_State)
workflow.set_entry_point("Intent_analyze")

workflow.add_node('Intent_analyze',Intent_analyze)
workflow.add_node('Router',Router)
workflow.add_node('Retrievers',Retrievers)
workflow.add_node('Reranker',Reranker)
workflow.add_node('_Web_search',_Web_search)
workflow.add_node('Web_searcher',Web_searcher)
workflow.add_node('Generater_dirctly',Generater_dirctly)
workflow.add_node('Generater',Generater)


workflow.add_edge('Router','Retrievers')
workflow.add_edge('Retrievers','Reranker')
workflow.add_edge('Reranker','Generater')
workflow.add_edge('Web_searcher','Generater')
workflow.add_edge('Generater',END)
workflow.add_edge('Generater_dirctly',END)

workflow.add_conditional_edges(
    'Intent_analyze',
    Retrieval_manager,
    {
        'Retrievers':'Router',
        'Web_search':'_Web_search'
    }
)

workflow.add_conditional_edges(
    '_Web_search',
    Web_search_manager,
    {
        'Web_searcher':'Web_searcher',
        'end':'Generater_dirctly',
    }
)

app = workflow.compile()