from langchain_core.runnables import (
    RouterRunnable,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSerializable
)
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# 初始化检索器
from PK_LLM_endfront.D_retriever import retrivers
import os
from PK_LLM_endfront.D_retriever.prompts import summary_prompt
from PK_LLM_endfront.D_retriever.prompts import answer_prompt,response_prompt
from operator import itemgetter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

class base_chain():
    def __init__(self) -> None:
        pass 
    def get_rag_chain(self,question:str = 'pfas')->RunnableSerializable:
        pass
    
    def answer_func(self,q:str)->str:
        rc = self.get_rag_chain(q) 
        answer = rc.invoke(q)
        return answer

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
class RagChain(base_chain):
    # 检索增强的chain
    def __init__(self,
                #IE:bool = True,
                #step_back:bool = False,
                #summary_llm = ChatOpenAI(),
                #answer_llm = ChatOpenAI(temperature=0.9),
                **kwargs
                ) -> None:
        
        IE = kwargs.get('IE',True)
        step_back = kwargs.get('step_back',True)
        summary_llm = kwargs.get('summary_llm',ChatOpenAI(temperature=0))
        answer_llm = kwargs.get('answer_llm',ChatOpenAI(temperature=0.9))
        rag_fusion:bool = kwargs.get('rag_fusion',True)
        r1:bool = kwargs.get('r1',True)
        r2:bool =  kwargs.get('r2',True)
        kb_id:int = kwargs.get('kb_id',1)
        vb_path:str =  kwargs.get('vb_path','retrievers')
        
        print(IE,step_back)
        _ = SubChain(
            rag_fusion= rag_fusion,
            r1= r1,
            r2= r2,
            kb_id= kb_id,
            vb_path= vb_path,
        )
        
        self.sub_chain = _.chain 
        self.retriever = _.retriever1
        self.summary_llm= summary_llm
        self.answer_llm=  answer_llm
        if IE and step_back:
            self.get_rag_chain = self.init_chain_IE_stepback
        elif IE and not step_back:
            self.get_rag_chain = self.init_chain_IE
        elif not IE and not step_back:
            self.get_rag_chain = self.init_clear_chain
        else:
            raise ValueError("paparment wrong")
        
    def summary_paral(self,input)->dict:
        # 返回一个可以交给RunnableParallel 的字典
        contexts = input['context']
        q = input['question']
        dir = {'question':RunnablePassthrough()}
        for i,_ in enumerate(contexts):
            def get_func(i):
                def f(input):
                    text = contexts[i]
                    return {'text':text,"question":q}
                return f
            dir[i] = get_func(i) | summary_prompt |self.summary_llm|StrOutputParser()
        return dir
    
    def init_chain_IE_stepback(self,question:str):
        # 初始化IE和step_back
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?",
            },
        ]
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge. 
                    Your task is to step back and paraphrase a question to a more generic step-back question, 
                    which is easier to answer. Here are a few examples:""",
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )
        generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
        
        # Response prompt 
        response_prompt_template = """You are an expert of world knowledge. 
            I am going to ask you a question. 
            Your response should be comprehensive and not contradicted with the following context 
            if they are relevant. Otherwise, ignore them if they are not relevant.
            * {text}
            * {step_back_context}

            * Original Question: {question}
            * Answer:"""
        response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
        _chain = self.sub_chain | self.summary_paral
        chains = _chain.invoke(question)
        rag_chain = (
            RunnableParallel(chains) # 信息提取
            | RunnableParallel( question = itemgetter("question") ,
                            text= self.format_context,
                            step_back_context = generate_queries_step_back|self.retriever,) 
            | response_prompt
            | self.answer_llm
            | StrOutputParser()
        )
        return rag_chain
    
    def init_chain_IE(self,question:str):
        # 初始化IE和step_back
        _chain = self.sub_chain | self.summary_paral
        chains = _chain.invoke(question)
        rag_chain = (
            RunnableParallel(chains) # 信息提取
            | RunnableParallel( question = itemgetter("question") ,text= self.format_context) 
            | answer_prompt
            | self.answer_llm
            | StrOutputParser()
        )
        return rag_chain
    
    def init_clear_chain(self,question:str):
        # not IE and not step_back
        # 初始化朴素检索增强chain
        _chain = self.sub_chain | self.summary_paral
        chains = _chain.invoke(question)
        rag_chain = (
            self.sub_chain
            | RunnableParallel( question = itemgetter("question") ,text= self.format_context) 
            | answer_prompt
            | self.answer_llm
            | StrOutputParser()
        )
        return rag_chain

    @staticmethod
    def format_context(input):
        text = ''
        for _ in input.keys():
            if _ != 'question':
                _text = input[_]
                text += f"\n {_text} \n"
        return text




class SubChain():
    lable = "用于双层检索的子链"
    chain:RunnableSerializable
    retriever1:VectorStoreRetriever = None
    def __str__(self) -> str:
        print(self.lable)
    def __init__(self,
                rag_fusion:bool = False,
                r1:bool = True,
                r2:bool = True,
                kb_id:int = 1,
                k1:int =5,
                k2:int = 10,
                vb_path:str = 'retrievers') -> None:
        self.db,self.db2 = retrivers.load_VBs(kb_id=kb_id,vb_path=vb_path)
        self.retriever1 = self.db.as_retriever(search_kwargs={"k": k1})  # 第一个检索器
        self.k2 = k2
        self.k1 = k1
        if rag_fusion and r1 and r2:
            chain = self.init_sub_chain_1()
        elif r1 and r2 and not rag_fusion:
            chain = self.init_sub_chain_2()
        elif r1 and not r2 and not rag_fusion: # 消除r2
            chain = RunnableParallel(question = RunnablePassthrough() ,context =self.retriever1) 
        elif r1 and not r2 and rag_fusion:
            chain = self.init_sub_chain_4()
        elif not r1 and r2 and not rag_fusion: # ablate r1
            chain = self.init_sub_chain_3()
        else:
            raise ValueError("paprament wrong")
        self.chain = chain
    
    def init_sub_chain_4(self):
        # 第一层检索加rag-fusion

        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        generate_queries = (
            prompt_rag_fusion 
            | ChatOpenAI(temperature=0)
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        
        from langchain.load import dumps, loads
        def reciprocal_rank_fusion(results: list[list], k=60)->list[Document]:
            """ 互惠排名融合方法，它接收多个排好序的文档列表
                以及在RRF公式中使用的可选参数k """
            fused_scores = {}
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    previous_score = fused_scores[doc_str]
                    # 使用RRF公式更新文档的得分：1 / (排名 + k)
                    fused_scores[doc_str] += 1 / (rank + k)
            reranked_results = [
                loads(doc)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            return reranked_results

        retrieval_chain_rag_fusion = generate_queries | self.retriever1.map() | reciprocal_rank_fusion
        
        sub_chain = (
            RunnableParallel( question=RunnablePassthrough(),context = retrieval_chain_rag_fusion)
        )
        return sub_chain
    
    
    def init_sub_chain_1(self)->RunnableSerializable:
        # 双层检索加Fusion
        # RAG-Fusion: Related
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        generate_queries = (
            prompt_rag_fusion 
            | ChatOpenAI(temperature=0)
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        
        from langchain.load import dumps, loads
        def reciprocal_rank_fusion(results: list[list], k=60)->list[Document]:
            """ 互惠排名融合方法，它接收多个排好序的文档列表
                以及在RRF公式中使用的可选参数k """
            fused_scores = {}
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    previous_score = fused_scores[doc_str]
                    # 使用RRF公式更新文档的得分：1 / (排名 + k)
                    fused_scores[doc_str] += 1 / (rank + k)
            reranked_results = [
                loads(doc)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            return reranked_results

        retrieval_chain_rag_fusion = generate_queries | self.retriever1.map() | reciprocal_rank_fusion
        
        sub_chain = (
            RunnableParallel( question=RunnablePassthrough(),docs1 = retrieval_chain_rag_fusion).assign(ids=self.get_ids) # 第一次检索
            | RunnableParallel(question = itemgetter('question') ,context =self.get_context)  # 进行第二次检索
        )
        return sub_chain

    def init_sub_chain_2(self):
        # 双层检索
        sub_chain = (
            RunnableParallel( question=RunnablePassthrough(),docs1 = self.retriever1).assign(ids=self.get_ids) # 第一次检索
            | RunnableParallel(question = itemgetter('question') ,context =self.get_context)  # 进行第二次检索
        )
        return sub_chain
    
    def init_sub_chain_3(self):
        retriever2 = self.db2.as_retriever( search_type="mmr",search_kwargs = {'k':self.k2,} )
        sub_chain = (
            RunnableParallel(question = RunnablePassthrough() ,context =retriever2) 
        )
        return sub_chain
    
    
    
    def get_context(self,input):
        question = input['question']
        ids = input['ids']
        retriever2 = self.db2.as_retriever( search_type="mmr",search_kwargs = {'k':self.k2, 'filter':{'doc_id':list(ids)} } )  # 第二个检索器
        output = retriever2.invoke(question)
        return output
    
    @staticmethod
    def get_ids(docs1:list[Document]):
        doc_ids = {doc.metadata['doc_id'] for doc in docs1['docs1']}  # 使用集合去重
        return doc_ids


class rag_chain(base_chain):
    # 默认双层检索加信息抽取的schain
    def __init__(self,kb_id:int = 1,vb_path:str = None) -> None:
        self.db,self.db2 = retrivers.load_VBs(kb_id=kb_id,vb_path=vb_path)
        self.retriever1 = self.db.as_retriever()  # 第一个检索器
        self.answer_llm = ChatOpenAI(model="gpt-3.5-turbo")  # 使用OpenAI的LLM
        self.summary_llm = ChatOpenAI(model="gpt-3.5-turbo")

    def get_context(self,input):
        question = input['question']
        ids = input['ids']
        retriever2 = self.db2.as_retriever( search_type="mmr",search_kwargs = {'k':10, 'filter':{'doc_id':list(ids)} } )  # 第二个检索器
        output = retriever2.invoke(question)
        return output
    
    @staticmethod
    def get_ids(docs1):
        doc_ids = {doc.metadata['doc_id'] for doc in docs1['docs1']}  # 使用集合去重
        return doc_ids

    def summary_paral(self,input):
        # 返回一个可以交给RunnableParallel 的字典
        contexts = input['context']
        q = input['question']
        dir = {'question':RunnablePassthrough()}
        for i,_ in enumerate(contexts):
            def get_func(i):
                def f(input):
                    text = contexts[i]
                    return {'text':text,"question":q}
                return f
            dir[i] = get_func(i) | summary_prompt |self.summary_llm|StrOutputParser()
        return dir

    def init_sub_chain(self):
        sub_chain = (
            RunnableParallel( question=RunnablePassthrough(),docs1 = self.retriever1).assign(ids=self.get_ids) # 第一次检索
            | RunnableParallel(question = itemgetter('question') ,context =self.get_context)  # 进行第二次检索
            | self.summary_paral
        )
        return sub_chain
    @staticmethod
    def format_context(input):
        text = ''
        for _ in input.keys():
            if _ != 'question':
                _text = input[_]
                text += f"\n {_text} \n"
        return text
        
    def get_rag_chain(self,question:str = 'pfas'):
        sub_chain = self.init_sub_chain()
        chains = sub_chain.invoke(question)
        rag_chain = (
            RunnableParallel(chains) # 信息提取
            | RunnableParallel( question = itemgetter("question") ,text= self.format_context) 
            | answer_prompt
            | self.answer_llm
            | StrOutputParser()
        )
        return rag_chain
    
    def answer_func(self,q:str)->str:
        rc = self.get_rag_chain(q) 
        answer = rc.invoke(q)
        return answer



# ***************************************


class RagChain_with_context(base_chain):
    # 检索增强的chain
    def __init__(self,
                **kwargs
                ) -> None:
        """可以返回检索结果的chain
        Raises:
            ValueError: _description_
        """
        if_answer = kwargs.get('if_answer',True)  # 是否输出回答，若False则只输出列表形式的检索内容
        IE = kwargs.get('IE',True) 
        step_back = kwargs.get('step_back',True)
        summary_llm = kwargs.get('summary_llm',ChatOpenAI(temperature=0))
        answer_llm = kwargs.get('answer_llm',ChatOpenAI(temperature=0.9))
        rag_fusion:bool = kwargs.get('rag_fusion',True)
        r1:bool = kwargs.get('r1',True)
        r2:bool =  kwargs.get('r2',True)
        kb_id:int = kwargs.get('kb_id',1)
        vb_path:str =  kwargs.get('vb_path','retrievers')
        k2:int =  kwargs.get('k2',10)
        k1:int =  kwargs.get('k1',5)
        _ = SubChain(
            rag_fusion= rag_fusion,
            r1= r1,
            r2= r2,
            kb_id= kb_id,
            vb_path= vb_path,
            k2 =k2,
            k1=k1
        )
        self.Subchain = _
        self.sub_chain = _.chain 
        self.retriever = _.retriever1
        self.summary_llm= summary_llm
        self.answer_llm=  answer_llm
        if not if_answer:
            self.get_rag_chain = self.init_chain_IE_stepback_but_context
        elif IE and step_back:
            self.get_rag_chain = self.init_chain_IE_stepback
        elif IE and not step_back:
            self.get_rag_chain = self.init_chain_IE
        elif not IE and not step_back:
            self.get_rag_chain = self.init_clear_chain
        elif not IE and step_back:
            self.get_rag_chain = self.init_chain_stepback
        else:
            raise ValueError("paparment wrong")
        
    def answer_func(self,q:str):
        rc = self.get_rag_chain(q) 
        # _ ={'output','context'}
        _ = rc.invoke(q)
        return _

    def summary_paral(self,input)->dict:
        # 返回一个可以交给RunnableParallel 的字典
        contexts = input['context']
        q = input['question']
        dir = {'question':RunnablePassthrough()}
        for i,_ in enumerate(contexts):
            def get_func(i):
                def f(input):
                    text = contexts[i]
                    return {'text':text,"question":q}
                return f
            dir[i] = get_func(i) | summary_prompt |self.summary_llm|StrOutputParser()
        return dir
    
    
    def init_chain_IE_stepback_but_context(self,question:str):
        # 初始化IE和step_back
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?",
            },
        ]
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge. 
                    Your task is to step back and paraphrase a question to a more generic step-back question, 
                    which is easier to answer. Here are a few examples:""",
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )
        generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()


        _chain = self.sub_chain | self.summary_paral
        chains = _chain.invoke(question)
        def format_context(input):
            context = []
            for _ in input.keys():
                if _ != 'question':
                    _text = input[_]
                    context.append(_text)
            print(f"length context = {len(context)}")
            return context
        rag_chain = (
            RunnableParallel(chains) # 信息提取
            | RunnableParallel( question = itemgetter("question") ,
                            context= format_context,
                            ) 
        )
        return rag_chain
    
    def init_chain_IE_stepback(self,question:str):
        # 初始化IE和step_back
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?",
            },
        ]
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge. 
                    Your task is to step back and paraphrase a question to a more generic step-back question, 
                    which is easier to answer. Here are a few examples:""",
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )
        generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
        

        _chain = self.sub_chain | self.summary_paral
        chains = _chain.invoke(question)
        def concat(input):
            t = input['text']
            for _ in input['step_back_context']:
                t+=_.page_content
            return t
        rag_chain = (
            RunnableParallel(chains) # 信息提取
            | RunnableParallel( question = itemgetter("question") ,
                            text= self.format_context,
                            step_back_context = generate_queries_step_back|self.retriever) 
            | { "context":concat  ,"output":response_prompt | self.answer_llm |StrOutputParser()}
        )
        return rag_chain

    def init_chain_stepback(self,question:str):
        # 初始化IE和step_back
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?",
            },
        ]
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge. 
                    Your task is to step back and paraphrase a question to a more generic step-back question, 
                    which is easier to answer. Here are a few examples:""",
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )
        generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
        def concat(input):
            t = input['text']
            for _ in input['step_back_context']:
                t+=_.page_content
            print(f"length context = {len(t)}")
            return t
        rag_chain = (
            self.sub_chain
            | RunnableParallel( question = itemgetter("question") ,
                            text= self.format_context,
                            step_back_context = generate_queries_step_back|self.retriever) 
            | { "context":concat  ,"output":response_prompt | self.answer_llm |StrOutputParser()}
        )
        return rag_chain
    def init_chain_IE(self,question:str):
        # 初始化IE和step_back
        _chain = self.sub_chain | self.summary_paral
        chains = _chain.invoke(question)
        rag_chain = (
            RunnableParallel(chains) # 信息提取
            | RunnableParallel( question = itemgetter("question") ,text= self.format_context) 
            | { "context":itemgetter('text')   ,"output":answer_prompt | self.answer_llm |StrOutputParser()}
        )
        return rag_chain
    
    def init_clear_chain(self,question):
        # 初始化朴素检索增强chain
        rag_chain = (
            self.sub_chain 
            | RunnableParallel( question = itemgetter("question") ,text= self.format_context) 
            | { "context":itemgetter('text')   ,"output":answer_prompt | self.answer_llm |StrOutputParser()}
        )
        return rag_chain

    @staticmethod
    def format_context(input):
        text = ''
        for _ in input.keys():
            if _ != 'question':
                _text = input[_]
                text += f"\n {_text} \n"
        print(f"length context = {len(text)}")
        return text