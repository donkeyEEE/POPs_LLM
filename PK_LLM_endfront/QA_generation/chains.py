from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from PK_LLM_endfront.QA_generation.prompts import Q_prompt,RAG_prompt
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class schema(BaseModel):
    Question:List[str]= Field(description="Questions")
        
def make_Q_chain(Q_prompt:str=Q_prompt,
                llm =ChatOpenAI(temperature=0.5))->schema:
    """生成问题的Chain
        结构化输出结果：Question:List[str]
    Args:
        Q_prompt (str, optional): ["title","abstract"]
        llm (_type_, optional): _description_. Defaults to ChatOpenAI(temperature=0.5).

    Returns:
        _type_: _description_
    """
    prompt = PromptTemplate(
        input_variables=["title","abstract"],
        template=Q_prompt
    )

        
    QA_Gen_Chain = prompt | llm .with_structured_output(schema)
    return QA_Gen_Chain


class schema2(BaseModel):
    Answer:str = Field(description="Your answer")
    Source_context:str = Field(description="原文来源")
    
def make_RAG_chain(pdf_path:str,
                llm  =ChatOpenAI(model='gpt-3.5-turbo',temperature=0.9),
                RAG_prompt:str = RAG_prompt)->schema2:
    """检索增强回答
    Answer:str = Field(description="Your answer")
    Args:
        pdf_path (str): pdf文献位置
        llm (_type_, optional): _description_. Defaults to ChatOpenAI(temperature=0.5).
        RAG_prompt (str, optional): ["context","question"] Defaults to RAG_prompt.

    Returns:
        _type_: _description_
    """
    loader = PyPDFLoader(pdf_path)
    docs  = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    prompt = PromptTemplate(
        input_variables=["context","question"],
        template=RAG_prompt
    )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm.with_structured_output(schema2)
    )
    return rag_chain


class schema3(BaseModel):
    result:List[int]= Field(description="QA的分数")

def check_QA(qa_lis:list[dir],llm =ChatOpenAI(temperature=0))->schema3:
    qas = ""
    for i,_ in enumerate(qa_lis):
        qas+=f"Question{i}: {_.get('Question')} \n Answer{i}: {_.get('Answer')} \n **** \n "
    
    prompt = PromptTemplate(
        input_variables=["qas"],
        template="""
            TASK: 给以下三对问答对，进行满分为10分的评分，给出整数的评价，分数越高，质量越高
            ***
            {qas}
            ***
        """
    )
    check_chain = prompt | llm .with_structured_output(schema3)
    return check_chain.invoke({"qas":qas})