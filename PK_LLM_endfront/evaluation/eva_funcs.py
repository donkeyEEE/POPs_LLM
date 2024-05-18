import json
import os
import random
import time
from typing import Dict, Any, List,Iterator
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from langsmith import Client
from PK_LLM_endfront.evaluation.utils import beautify_output
from PK_LLM_endfront.evaluation.llms import predictor, PredictorFactory,RAGPredictor_with_context
from ragas.metrics import answer_correctness, answer_similarity,answer_relevancy
from ragas import evaluate
import ast

"""
测试类demo
# 合并多个QA的json文件
>>> from PK_LLM_endfront.evaluation.eva_funcs import Dataloader
>>> dl = DataLoader()
>>> paths = [f'_data\QA\good_qa_data0{i+1}.json' for i in range(4)]
>>> all_data_path = "_data/QA/goog_1.json"
>>> dl.merge_json_files(paths,output_file=all_data_path)


# 运行或者重复运行测试流程
>>> manager = Manager(model='glm-4',filepath="eval_reasult/1/eval")
>>> manager.restart_predict(dataloader)

# 调用mannager进行预测并且评估
for i in range(1,7):
    if i < 1:
        continue
    dataloader = DataLoader(f"_data\\testqa\\testdata_{i}.json")
    dataloader.data = dataloader.load_json(dataloader.filepath)
    #dataloader.data = dataloader.load_json(dataloader.filepath)
    manager = Manager(model='rag',filepath=f"eval_reasult/{i}/eval")
    #manager.restart_predict(dataloader)
    manager.run_predict(dataloader)
    manager.run_evaluate()
    manager.save()
"""

'''
# manager类预测原理
import os
os.chdir('E:\学习\python\py_codbase\PK_LLM')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from PK_LLM_endfront.evaluation.eva_funcs import DataLoader,Evaluator,Record
from PK_LLM_endfront.evaluation.llms import predictor,PredictorFactory,GPTPredictor
from pprint import pprint
dataloader = DataLoader(f"_data\\testqa\\testdata_{1}.json")
e = Evaluator()
# 1. 获取Record类
r = next(dataloader.get_data()) # record demo
print(r)
# 2. make Predictor instance
# 2.1 use PredictorFactory 
p = PredictorFactory().get_predictor(model_type='gpt-3.5-turbo',
                                    temperature = 0.9)
print(type(p))
# 2.2 creat dirctly
p = GPTPredictor()
p.run("我应该如何准备和女朋友的纪念日")

# 3.Put Predictor into Record to execute example
r.add_prediction(p)
print(r)

# 4. To perform batch testing, integrate the current record into the evaluator.
e.add_Record(r)
e.evaluate()

# or Evaluator provide a method to eun eval easily
# e.run_eval(dataloader=dataloader,model="gpt-3.5-turbo",time_=0.2)

# 5. save result 
e.save(f"_temp/eval_{p.prefix}.csv")
e.df
'''

class smith_client():
    """用于连接smith的工具
    """
    def __init__(self,dataset_name = 'QA_TEST_1') -> None:
        self.client = Client()
        self.dataset_name = dataset_name
        self.beautify_output = beautify_output
    def show_dataset(self):
        for _ in self.client.list_datasets():
            print(_)
    def creat_upload_data(self,path):
        dataset = self.client.create_dataset(dataset_name=self.dataset_name)
        data = self.load_extract_data(path)
        for _i,_ in enumerate(data):
            self.client.create_examples(
                inputs=[
                    {"question":_["Question"]}
                ],
                outputs=[
                    {"answer":_["Answer"]}
                ],
                metadata=[{"id":_i,"doi":_["DOI"],'source_context':_["Source_context"],'time':_['time']}],
                dataset_id=dataset.id
            )
    def upload_data(self,path):
        # dataset = self.client.create_dataset(dataset_name=self.dataset_name)
        data = self.load_extract_data(path)
        for _i,_ in enumerate(data):
            self.client.create_examples(
                inputs=[
                    {"question":_["Question"]}
                ],
                outputs=[
                    {"answer":_["Answer"]}
                ],
                metadata=[{"id":_i,"doi":_["DOI"],'source_context':_["Source_context"]}],
                dataset_name=self.dataset_name
            )
    @staticmethod
    def load_json(path)->dict:
        # 打开并读取JSON文件
        with open(path, 'r', encoding='utf-8') as file:
            # 将JSON内容转换为字典
            data = json.load(file)
        return data

    @staticmethod
    def merge_json_files(file_paths, output_file=None):
        """
        Merges multiple JSON files into a single list. Each file should contain a list of dictionaries.
        
        # Example usage:
        # files_to_merge = ['path/to/file1.json', 'path/to/file2.json', ...]
        # merged_data = merge_json_files(files_to_merge, 'path/to/output.json')
        """
        merged_data = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    merged_data+= data
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        if output_file:
            try:
                with open(output_file, 'w') as file:
                    json.dump(merged_data, file, indent=4)
            except Exception as e:
                print(f"Error writing to {output_file}: {str(e)}")
        return merged_data


    @staticmethod
    def load_extract_data(file_path, num_samples=50):
        # 加载JSON数据
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        results = []
        included_years = set()  # 用于跟踪已包含的年份
        e = []
        # 确保每年至少有一个条目
        for entry in data:
            if entry["time"] not in included_years:
                results.append(entry)
                included_years.add(entry["time"])
                e.append(entry['DOI'])
        
        # 计算需要的额外条目数量
        additional_needed = num_samples - len(results)

        # 从剩余条目中随机抽取
        additional_entries = [entry for entry in data if entry["DOI"] not in e]
        random.shuffle(additional_entries)  # 随机排列

        # 添加所需数量的额外条目
        results.extend(additional_entries[:additional_needed])

        # 返回前50个条目
        return results[:num_samples]



class Record:
    """原始QA数据的抽象
    {
        "Question": "What are the implications of climate change on cardiovascular disease?",
        "Answer": "Climate change can adversely affect cardiovascular health. The health outcomes are diverse and complex and include several exposure pathways that might promote the development of non-communicable diseases such as cardiovascular disease. The consequences of climate change on cardiovascular health result from direct exposure pathways, such as shifts in ambient temperature, air pollution, forest fires, desert (dust and sand) storms and extreme weather events. Direct exposure to extreme weather events, ambient temperatures, heat waves, cold spells and a wide array of pollutants has the potential to exacerbate disease in individuals with underlying cardiovascular conditions and contribute to the development of disease in those without known cardiovascular disease.",
        "Source_context": "Climate change is the greatest existential threat to humans and can adversely affect cardiovascular health. Health outcomes are diverse and complex and include several exposure pathways that might promote the development of non-communicable diseases such as cardiovascular disease. In this Review, we aim to provide an overview of the consequences of climate change on cardiovascular health, which result from direct exposure pathways, such as shifts in ambient temperature, air pollution, forest fires, desert (dust and sand) storms and extreme weather events. Direct exposure to extreme weather events, ambient temperatures, heat waves, cold spells and a wide array of pollutants has the potential to exacerbate disease in individuals with underlying cardiovascular conditions and contribute to the development of disease in those without known CVD.",
        "DOI": "10.1038/s41569-022-00720-x",
        "time": "2022",
        "Score": 10
    },
    """
    def __init__(self, data: dict):
        self.data = data  # 存储原始数据
        self.prediction:str = None  # 用于存储预测结果
        self.evaluation:Dict = None  # 用于存储评估结果
        self.context = None

    def add_prediction(self, predictor: predictor):
        """添加预测结果到记录中"""
        if isinstance(predictor,RAGPredictor_with_context):
            self.prediction ,self.context = predictor.run(self.data['Question'])
        else:
            self.prediction = predictor.run(self.data['Question'])

    def add_evaluation(self, evaluation: dict):
        """添加评估结果到记录中"""
        self.evaluation = evaluation

    def __str__(self):
        """方便打印和查看记录的内容"""
        return f"Record(Data={self.data.keys()}, Prediction={self.prediction}, Evaluation={self.evaluation})"


class DataLoader:
    def __init__(self, filepath: str =None):
        self.filepath = filepath
        self.data = None  # 初始化数据为None，使用懒加载

    def get_data(self) -> Iterator[Record]:
        """
        从JSON文件懒加载数据，并逐条返回。
        loader = DataLoader('path_to_good_qa_data00.json')
        for record in loader.get_data():
            print(record)  # 处理每条记录
        """
        if self.data is None:
            self.data = self.load_json(self.filepath)
        for _ in self.data:
            yield(Record(_))

    @staticmethod
    def load_json(path: str) -> List[Dict[str, Any]]:
        """
        加载JSON文件并返回数据列表。
        """
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"The file {path} was not found.")
            return []  # 返回空列表作为安全措施
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {path}.")
            return []  # 返回空列表作为安全措施

    @staticmethod
    def merge_json_files(file_paths: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """
        合并多个JSON文件到一个列表，并可选地写入到一个新的JSON文件。
        """
        merged_data = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    merged_data.extend(data)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(merged_data, file, indent=4)
            except IOError:
                print(f"Error writing to {output_file}.")
        return merged_data

    @staticmethod
    def load_extract_data(file_path: str, num_samples: int = 50) -> List[Dict[str, Any]]:
        """
        从指定的JSON文件加载并提取特定数量的样本。
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}.")
            return []

        results = []
        included_years = set()
        doi_included = set()
        for entry in data:
            if entry["time"] not in included_years:
                results.append(entry)
                included_years.add(entry["time"])
                doi_included.add(entry['DOI'])
        
        additional_needed = num_samples - len(results)
        additional_entries = [entry for entry in data if entry["DOI"] not in doi_included]
        random.shuffle(additional_entries)
        results.extend(additional_entries[:additional_needed])

        return results[:num_samples]



class Evaluator():
    record_lis:List[Record] = []
    df:pd.DataFrame = None
    def __init__(self) -> None:
        pass
    
    def add_Record(self,record:Record):
        self.check_record(record)
        self.record_lis.append(record)

    def evaluate(self,
                metrics:List = [answer_correctness,answer_similarity]
                ):
        data_samples = {
            'question':[],
            'answer':[],
            'ground_truth':[]
        }
        
        for r in self.record_lis:
            data_samples['question'].append(r.data.get("Question"))
            data_samples['answer'].append(r.prediction)
            data_samples['ground_truth'].append(r.data.get("Answer"))
        
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(dataset,metrics=metrics)
        df = score.to_pandas()
        
        for i,_i in enumerate(df.index):
            self.record_lis[i].evaluation = {'answer_correctness':df['answer_correctness'].loc[_i],
                                            "answer_similarity" :df['answer_similarity'].loc[_i]
                                            }
            
    
    def save(self, filepath: str):
        # 创建一个空的DataFrame
        data = {
            'Question': [],
            'Answer': [],
            'answer_correctness': [],
            'answer_similarity':[],
            'DOI': [],
            'Time': []
        }

        # 遍历记录列表，填充数据
        for record in self.record_lis:
            data['Question'].append(record.data.get('Question', ''))
            data['Answer'].append(record.prediction if record.prediction else '')
            # 假设评估结果是字典形式且可以直接转换为字符串
            data['answer_correctness'].append(record.evaluation['answer_correctness'])
            data['answer_similarity'].append(record.evaluation['answer_similarity'])
            data['DOI'].append(record.data.get('DOI', ''))
            data['Time'].append(record.data.get('time', ''))

        # 创建DataFrame
        df = pd.DataFrame(data)
        # 保存DataFrame到CSV
        df.to_csv(filepath, index=False)
        self.df = df

    def run_eval(self,dataloader:DataLoader,model="gpt-4",time_=1.0,**kwargs):
        """_summary_

        Args:
            dataloader (DataLoader): _description_
            model (str, optional): _description_. Defaults to "gpt-4".
            time_ (float, optional): 预测器运行间隔，用于规避API调用速率问题. Defaults to 1.0.
        """
        predictor = PredictorFactory.get_predictor(model,**kwargs)
        for _i ,record in tqdm(enumerate(dataloader.get_data())):
            record.add_prediction(predictor)
            self.add_Record(record)
            time.sleep(time_)
        self.evaluate()
        self.save(f"eval_{model}.csv")
        
        #eva_result = self.evaluate(record)
        #record.add_evaluation(eva_result)
    @staticmethod
    def check_record(record:Record):
            if not isinstance(record.data, dict):
                raise TypeError("Record data is expected to be a dictionary.")
            if record.prediction is None:
                raise ValueError("Record is missing a prediction.")


class Manager():
    record_lis:list[Record] = []
    filepath = "tmp.csv"
    df:pd.DataFrame = None
    def __init__(self,model="gpt-4",filepath="tmp",**kwargs) -> None:
        """测试管理器

        Args:
            model (str, optional): 模型代号:gpt-4 ; gpt-3.5-turbo ; glm-3-turbo ; glm-4 ; rag;rag_context
            filepath (str, optional): 结果文件储存位置，也会被用来二次加载
        """
        self.predictor = PredictorFactory.get_predictor(model,**kwargs)
        self.filepath =f"{filepath}_{model}.csv" 
        
    def run_predict(self,dataloader:DataLoader,predictor:predictor =None):
        if predictor is None:
            predictor = self.predictor
        for _i ,record in enumerate(dataloader.get_data()):
            print(_i)
            record.add_prediction(predictor)
            self.record_lis.append(record)
            
    def restart_predict(self,dataloader:DataLoader):
        if len(self.record_lis)==0:
            try:
                self.load_from_csv()
            except:
                pass
        for _i ,record in enumerate(dataloader.get_data()):
            print(_i)
            if _i <len(self.record_lis):
                continue
            record.add_prediction(self.predictor)
            self.record_lis.append(record)
            self.save()
    
    def run_evaluate(self,
                metrics:List = [answer_correctness,answer_similarity]
                ):
        data_samples = {
            'question':[],
            'answer':[],
            'ground_truth':[]
        }
        
        for r in self.record_lis:
            data_samples['question'].append(r.data.get("Question"))
            data_samples['answer'].append(r.prediction)
            data_samples['ground_truth'].append(r.data.get("Answer"))
        
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(dataset,metrics=metrics)
        df = score.to_pandas()
        
        for i,_i in enumerate(df.index):
            self.record_lis[i].evaluation = {'answer_correctness':df['answer_correctness'].loc[_i],
                                            "answer_similarity" :df['answer_similarity'].loc[_i]
                                            }
    
    def save(self):
        # 创建一个空的DataFrame
        data = {
            'data': [],
            'prediction': [],
            'Question': [],
            'answer_similarity':[],
            'answer_correctness':[]
        }
        _c =[] 
        for record in self.record_lis:
            data['data'].append(str(record.data))
            data['Question'].append(record.data.get('Question', ''))
            data['prediction'].append(record.prediction if record.prediction else '')
            
            if record.context:
                _c .append(record.context)
            
            # 假设评估结果是字典形式且可以直接转换为字符串
            try:
                data['answer_correctness'].append(record.evaluation['answer_correctness'])# record.evaluation['answer_correctness']
                data['answer_similarity'].append(record.evaluation['answer_similarity'])# record.evaluation['answer_similarity']
            except:
                data['answer_correctness'].append("") 
                data['answer_similarity'].append("")
        if len(_c) !=0:
            # 有可能出问题
            data['context'] = _c
        # 创建DataFrame
        df = pd.DataFrame(data)
        # 保存DataFrame到CSV
        df.to_csv(self.filepath, index=False)
        self.df = df
        
    def save_csv(self):
        self.df.to_csv(self.filepath, index=False)
    
    def load_from_csv(self,filepath=None):
        if filepath == None:
            filepath = self.filepath
        df = pd.read_csv(filepath)
        r_lis = []
        for _index in df.index:
            str_data = df.loc[_index]['data']
            record = Record(data=ast.literal_eval(str_data))
            record.prediction = df.loc[_index]['prediction']
            record.evaluation = {'answer_similarity':df.loc[_index]['answer_similarity'],
                                'answer_correctness':df.loc[_index]['answer_similarity']}
            r_lis.append(record)
        self.record_lis = r_lis
