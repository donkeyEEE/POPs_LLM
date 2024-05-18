
# edges
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from PK_LLM_endfront.loggs import logger



def Retrieval_manager(input:dict)->str:
    logger.info('---Doing Retrieval_manager---')
    logger.info(input)
    if input['retrieval']:
        logger.info(f"{input['question'] } need retrieval")
        return  'Retrievers'
    else:
        logger.info(f"{input['question'] } don't need retrieval")
        return 'Web_search'

def Web_search_manager(input:dict)->str:
    """选择是否继续web search
    Args:
        input (dict): {'question':question_transform}
    """
    logger.info('---Doing Web_search_manager---')
    llm = ChatOpenAI()
    class if_r(BaseModel):
        retrieval:bool =Field(description='是否需要进行网络检索')
    prompt = PromptTemplate(
        input_variables=['question'],
        template="你是一个网络检索判断器，你需要判断要解决此问题，你是否需要进行有关网络的检索增强\n\n{question}"
    )
    
    chain = prompt |llm.with_structured_output(if_r)
    r = chain.invoke({'question':input['question']})
    if r.retrieval:
        return  'Web_searcher'
    else:
        return 'end'