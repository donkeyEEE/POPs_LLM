# nodes
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from PK_LLM_endfront.loggs import logger


def get_intent_analyze_chain()->Runnable:
    llm = ChatOpenAI()
    template = """
您是一个问题重写器，可以结合对话历史将输入问题转换为一个更好的版本，这个版本针对向量存储检索进行了优化。 
查看并尝试推理其背后的语义意图/含义。
并且请你分析要回答此问题你是否需要额外的信息
***
用户问题：{question}
对话历史: {history}
"""
    class intent_output(BaseModel):
        retrieval:bool = Field(description="是否需要额外信息，已用于处理复杂的问题，若此问题你可以自行解决则不需要额外信息")
        question:str = Field(description="优化版本的问题")
    
    prompt = PromptTemplate(
        input_variables=['question','history'],
        template=template
    )
    return prompt|llm.with_structured_output(intent_output)
    
def Intent_analyze(input:dict)->dict:
    """分析用户需求
    Args:
        input (dict): {'question','history'}
    Returns:
        dict: {'retrieval':bool,'question':question_transform}
    """
    logger.info("---Doing Intent_analyze---")
    _Chain = get_intent_analyze_chain()
    question = input.get('question','')
    history = input.get('history','')
    output = _Chain.invoke({'question':question,'history':history})
    logger.info(output)
    logger.info(f'{question} has been transformed into {output.question}')
    
    return {'retrieval':output.retrieval,'question':question} # output.question
"""
r= Intent_analyze({'question':'你是谁','history':""})
logger.info(r)
"""


import random
def get_router_chain()->Runnable:
    llm = ChatOpenAI(temperature=0.9)
    template = """
你是一个能将用户问题路由到不同知识库的专家，你将分析用户问题，
并且给出应该从哪些知识库中检索数据，注意你的输出是一个列表，其中元素为代号
选择标准：
* 需要尽可能少的选择需要的知识库
* 注意你的输出的数量控制在三个以内，比如[1,2,4]
***
问题：{question}

知识库代号和内容的对应关系如下：
1  ---》 Analytical_Methods:分析方法
2  ---》 Environmenta_Exposure：环境暴露
3  ---》 Environmental_Behavior：环境行为
4  ---》 Biological_Behavior：生物行为
5  ---》 Toxicity：毒性
6  ---》 Health Risk：人类风险


***

"""
    class router_output(BaseModel):
        domain:List[int] = Field(description="需要使用到的知识库列表，元素为知识库代号，注意你的输出的数量控制在三个以内")
    
    prompt = PromptTemplate(
        input_variables=['question'],
        template= template
    )
    return prompt|llm.with_structured_output(router_output)

def Router(input:dict)->dict:
    """检索任务分发
    Args:
        input (dict): {'question':question_transform}
    Returns:
        dict: {'domain':List[1,2,3,4,5,6]}
    """
    logger.info('---Doing Router---')
    elements = [1, 2, 3, 4, 5, 6]
    _d = random.sample(elements, 2)
    logger.info(f"选择的知识库代号为：{_d}")
    return {'domain':[1]}
    """
    _chain = get_router_chain()
    question = input.get('question','')
    output = _chain.invoke({'question':question})
    logger.info(f"选择的知识库代号为：{output.domain}")
    return {'domain':output.domain}
    """
    
"""
r = Router({'question':'如何解决PFAS污染问题'})
logger.info(r)
"""


from PK_LLM_endfront.D_retriever.chains import RagChain_with_context
def Retrievers(input:dict)->dict:
    """选择性进行双层检索
    Args:
        input (dict): {'question':question_transform,'domain':List[1,2,3,4,5,6]}
    Returns:
        dict: {'context':List[str]}
    """
    logger.info('---Doing Retrievers---')
    context = []
    for _ in input.get('domain',''):
        if _ in [1,2,3,4,5,6]:
            _ragchain = RagChain_with_context(kb_id=_,if_answer = False,r2=False,IE=False,rag_fusion=False)
        else:
            logger.warning(f"get wrong domain :'{_}'")
            continue
        r = _ragchain.answer_func(input.get('question',''))
        logger.info(f"domain {_} get {len(r['context'])} context")
        context+=r['context']
    return {'context':context}
"""
r= Retrievers({'question':'如何解决PFAS污染','domain':[1,2,4]})
logger.info(r)
"""



import random
def Reranker(input:dict)->dict:
    """对多个问答进行打分并且重排
    Args:
        input (dict):{'context':List[str],'question':question_transform}
    Returns:
        dict: {'context':List[str]}
    """
    logger.info('---Doing Reranker---')
    # 获取context列表
    context_list = input.get('context', [])
    # 确保context列表不为空
    if not context_list:
        raise ValueError("Context list is empty.")
    # 随机选择不超过10个元素
    selected_context = random.sample(context_list, min(10, len(context_list)))
    # 返回结果
    return {'context': selected_context}
"""
r= Reranker({'context':[i for i in range(20)]})
len(r['context'])
"""

### Search

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

def Web_searcher(input:dict)->dict:
    """网络搜索工具
    Args:
        input (dict): {'question':question_transform}

    Returns:
        dict: {'question':question_transform,'context':List[str]}
    """
    logger.info("---WEB SEARCH---")
    question = input["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = [d["content"] for d in docs]
    return {"context": web_results, "question": question}
"""
r = Web_searcher({'question':'什么是PFAS'})
r
"""


from langchain_core.output_parsers import StrOutputParser
def Generater(input:dict)->dict:
    """基于信息回答问题
    Args:
        input (dict): {'context':List[str],'question':question_transform}

    Returns:
        dict: {'context':List[str],'question':question_transform,'output':generation}
    """
    logger.info('---Doing Generater---')
    context = input['context']
    _context ='\n'.join(context)
    
    template = """
作为化学专家，你按照以下详细步骤回答问题：

1. **分析问题**：仔细阅读问题，理解其关键点和所需信息的类型。识别相关的化学和环境科学知识，并考虑问题的背景。

2. **审查文档内容**：仔细阅读并且理解提供的文档，从中寻找相关的事实、数据和证据。注意关键细节，确保信息准确。

3. 根据文档内容和提取的证据，创建一个基于化学原理的连贯且详细的回答。确保答案逻辑清晰，并考虑可能的替代解释。

4. **处理证据不足**：如果文档中没有足够的证据，明确说明无法回答或解释该问题。避免猜测，并建议可能的下一步，如进一步研究或数据收集。

***
问题：{question}
***
文档内容：
{context}
***
**注意事项**
- 确保每一步都基于证据，避免无根据的假设。
- 使用专业的化学术语，确保答案清晰易懂。

"""
    
    prompt = PromptTemplate(
        input_variables=['question','context'],
        template=template
    )
    llm = ChatOpenAI(temperature=0.9)
    
    _chain = prompt | llm |StrOutputParser()
    r = _chain.invoke({'question':input['question'],'context':_context})
    return {'output':r}

def Generater_dirctly(input:dict)->dict:
    """直接回答问题
    Args:
        input (dict): {'question':question_transform}

    Returns:
        dict: {'question':question_transform,'output':generation}
    """
    logger.info('---Doing Generater_dirctly---')
    llm = ChatOpenAI()
    chain = llm |StrOutputParser()
    r = chain.invoke(input['question']) 
    return {'output':r}



def _Web_search(input):
    return input