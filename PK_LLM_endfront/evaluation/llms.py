
from langchain_openai import ChatOpenAI
import time
from zhipuai import ZhipuAI

# 工厂函数
class PredictorFactory:
    @staticmethod
    def get_predictor(model_type: str, **kwargs):
        if 'gpt' in model_type:
            return GPTPredictor(**kwargs)
        elif 'glm' in model_type:
            return GLMPredictor(**kwargs)
        elif model_type  == 'rag' :
            return RAGPredictor(**kwargs)
        elif 'rag_context' in model_type:
            return RAGPredictor_with_context(**kwargs)
        else:
            raise ValueError("Unknown model type provided")

# 预测器基类
class predictor():
    predict_func = None
    prefix = None
    def func(self):
        # 预测函数对象
        return self.predict_func
    
    def run(self, query: str) -> str:
        # 能直接运行的方法
        if not self.predict_func:
            raise ValueError("Predict function not set.")
        inputs = {'question': query}

        try:
            response = self.predict_func(inputs)
            return response.get("output", "")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return ""

from PK_LLM_endfront.D_retriever import chains

class RAGPredictor(predictor):
    def __init__(self,**kwargs) -> None:
        super().__init__()
        self.prefix = 'Advance RAG'
        # 替换为新的链
        self.chain = chains.RagChain(**kwargs)
        
    def predict_func(self,inputs:dict):
        r = self.chain.answer_func(inputs["question"])
        return {"output": r}

class RAGPredictor_with_context(predictor):
    def __init__(self,**kwargs) -> None:
        super().__init__()
        self.prefix = 'Advance RAG'
        # 替换为新的链
        self.chain = chains.RagChain_with_context(**kwargs)
        
    def predict_func(self,inputs:dict):
        r = self.chain.answer_func(inputs["question"])
        return {"output": r['output'],'context':r['context']}
    
    def run(self, query: str) -> str:
        # 能直接运行的方法
        if not self.predict_func:
            raise ValueError("Predict function not set.")
        inputs = {'question': query}

        try:
            response = self.predict_func(inputs)
            return response.get("output", "") ,response.get("context", "")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return ""

class GPTPredictor(predictor):
    def __init__(self,model:str = "gpt-3.5-turbo",temperature=0.5) -> None:
        super().__init__()
        self.llm = self.init_llm(model = model,temperature=temperature)
        self.predict_func = self.get_func(self.llm)
        if model == 'gpt-3.5-turbo':
            self.prefix = "eva_gpt3.5"
        if model == 'gpt-4':
            self.prefix = "eva_gpt4"
    @staticmethod
    def get_func(llm):
        def func(inputs:dict):
            _f = 0
            while _f < 5:
                try:
                    response = llm.invoke(inputs["question"])
                    _f =10 
                except Exception as e:
                    print(e)
                    # 避免Openai的速率限制
                    _f +=1
                    time.sleep(10)
            return {"output": response.content}
        return func
        
    @staticmethod
    def init_llm(model:str,temperature=0.5):
        return ChatOpenAI(model=model,temperature=temperature)

class GLMPredictor(predictor):
    def __init__(self,model:str = "glm-3-turbo",temperature=0.5) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature
        try:
            self.predict_func =self.get_func()
            self.prefix = self.model
        except Exception as e:
            print(e)
    def get_func(self):
        if self.model not in ['glm-3-turbo','glm-4']:
            raise ValueError(f"The model \"{self.model}\" is not available!!!")
        
        def pre_fun(inputs: dict) -> dict:
            client = ZhipuAI(api_key="eb2b404fd0ceaa802d2be3d91a73fc3b.m1Aq1vtQpn6T6fAM") 
            response = client.chat.completions.create(
                model=self.model,  # 填写需要调用的模型名称
                messages=[
                    {"role": "user", "content": inputs.get("question")}
                ],
                temperature = self.temperature
            )
            return {"output": response.choices[0].message.content}
        return pre_fun