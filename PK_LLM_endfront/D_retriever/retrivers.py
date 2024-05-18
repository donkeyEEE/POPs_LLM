import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever,VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

class vb_mannger():
    def __init__(self,data_path:str = "_data/demo") -> None:
        self.data_path = data_path # 数据集路径
        try:
            self.df = pd.read_csv(f"{data_path}/litera_lis.csv",encoding="ISO-8859-1",index_col=0)
        except:
            self.df = pd.read_csv(f"{data_path}/litera_lis.csv",index_col=0)
    
    def load_pdf(self,doc_id: str= 'SL01-21-J' ):
        pdf_path = f"{self.data_path}/literature_PDF\{doc_id}.pdf"
        loader = PyPDFLoader(pdf_path)
        docs  = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        for _ in splits:
            _.metadata['doc_id'] = doc_id
        return splits

    def creat_VBs(self):
        docs = []
        for _index in tqdm(self.df.index):
            _a = self.df['Abstract'].loc[_index]
            if type(_a) == type("sss") and _a != "No abstract available for this PMID." and \
                self.df['is_download'].loc[_index] ==1:
                print(_index)
                doc = Document(page_content=_a,metadata={"doc_id":_index})
                docs.append(doc)
        db = FAISS.from_documents(docs,OpenAIEmbeddings())
        docs = []
        for doc_id in tqdm(self.df.index):
            _a = self.df['Abstract'].loc[doc_id]
            if type(_a) == type("sss") and _a != "No abstract available for this PMID." and \
                self.df['is_download'].loc[doc_id] ==1:
                print(doc_id)
                try:
                    doc = self.load_pdf(doc_id)
                    docs+=doc
                except Exception as e:
                    print(f"{doc_id} error:{e}")
        db2 = FAISS.from_documents(docs,OpenAIEmbeddings())
        return db,db2
"""
data_path = "_data/demo"
df = pd.read_csv(f"{data_path}/litera_lis.csv",encoding="ISO-8859-1",index_col=0)

def load_pdf(doc_id: str= 'SL01-21-J' ):
    pdf_path = f"{data_path}/literature_PDF\{doc_id}.pdf"
    loader = PyPDFLoader(pdf_path)
    docs  = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    for _ in splits:
        _.metadata['doc_id'] = doc_id
    return splits


def creat_VBs(df:pd.DataFrame):
    docs = []
    for _index in df.index:
        doc = Document(page_content=df['Abstract'].loc[_index],metadata={"doc_id":_index})
        docs.append(doc)
    db = FAISS.from_documents(docs,OpenAIEmbeddings())
    docs = []
    for doc_id in df.index:
        doc = load_pdf(doc_id)
        docs+=doc
    db2 = FAISS.from_documents(docs,OpenAIEmbeddings())
    return db,db2
import os
"""
def load_VBs(kb_id:int = 1,vb_path:str = "retrievers")->VectorStore:
    path1 = f"{vb_path}/{kb_id}db1"
    path2 = f"{vb_path}/{kb_id}db2"
    db = FAISS.load_local(path1,OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    db2 = FAISS.load_local(path2,OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    return db,db2



