from langchain_core.prompts.chat import ChatPromptTemplate
from typing import List
from langsmith.schemas import Example, Run
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# 你是一个具有环境领域专业知识的学者，
# 请你判断answer中是否包括参考答案中基本科学事实
# 认为包含，则输出True，若不包含或者答案中有违反科学事实的地方，则返回False"

eva_prompt_str = """
You are a scholar with expertise in the environmental field. 
Please judge whether the answer includes the basic scientific facts in the reference answer. 
If you believe it does, output True. 
If it does not include the facts or if there are violations of scientific facts in the answer, return False.
"""
evaluator_prompt = ChatPromptTemplate.from_messages(
    [ 
        (
            "system",eva_prompt_str
            ),
        ("human","answer : {answer} \n reference answer : {re_answer}")
    ]
)



class Is_conforms(BaseModel):
    score:bool = Field(description="whether the answer includes the basic scientific facts in the reference answer. ")
llm = ChatOpenAI(temperature=0)
eva_chain = evaluator_prompt |llm.with_structured_output(schema=Is_conforms)
def faithfulness(run: Run, example: Example) -> dict: 
    answer:str = run.outputs["output"]
    re_answer:str = example.outputs["answer"]
    # question:str = example.inputs["question"]
    # score= eva_chain.invoke({"question":question,"long_answer":long_answer,"answer":answer})
    score= eva_chain.invoke({"re_answer":re_answer,"answer":answer})
    print( example.inputs["question"])
    return {"key": "correctness_2", "score": score.score} 

# eva_chain 测试
#\dataset = client.read_dataset(dataset_name=dataset_name)
#examples = list(client.list_examples(dataset_id=dataset.id))
#e = examples[0]
#r = eva_chain.invoke({"question":e.inputs,"long_answer":e.outputs,"answer":"我是你叠"})