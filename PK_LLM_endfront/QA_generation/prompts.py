Q_prompt_ = """
You are an exam creator specializing in environmental engineering.
TASK：Change the article title to a question
***
Note:
    1. 请先仔细阅读文献摘要，理解本研究的大致研究背景、方法和目标。
    2. 重新构思标题，将其改写为既准确又具体的问题
    3. 问题必须准确反映文章的研究内容，避免引入与研究主题无关的概念或误解。确保问题直接与文章的研究焦点和结论相对应。
    4. 请用中文，同时给出三个问题。
 ***
标题 :{title}
摘要:{abstract}
***
例子：
    * 标题：Serum Concentrations of Per- and Polyfluoroalkyl Substances and Risk of Renal Cell Carcinoma
    * 问题：PFAS血清浓度与肾细胞癌风险之间存在怎样的关联？
"""

Q_prompt = """
You are an exam creator specializing in environmental engineering.
TASK: Change the article title to a question

Note:
1. Please read the article abstract carefully to understand the general research background, methods, and objectives of the study.
2. Rethink the title and rewrite it as an accurate and specific question.
3. The question must accurately reflect the content of the article, avoiding the introduction of concepts or misunderstandings unrelated to the research topic. Ensure the question directly corresponds to the research focus and conclusions of the article.
4. Please provide three questions.

Title: {title}
Abstract: {abstract}

Example:
* Title: Serum Concentrations of Per- and Polyfluoroalkyl Substances and Risk of Renal Cell Carcinoma
* Question: What is the association between PFAS serum concentrations and the risk of renal cell carcinoma?
"""

RAG_prompt = """
As a respondent with expertise in environmental chemistry, you are tasked with analyzing the following dataset text. Your insights will contribute to a reference dataset, guiding future analysis and understanding in the field. Please adhere to the following guidelines when crafting your response:
- Focus on delivering a precise and factual answer based on the text.
- Avoid using superfluous filler words or phrases that do not contribute to the clarity or accuracy of your response.
- Construct your response by following a logical chain of reasoning that connects the question to the relevant evidence in the text.

To ensure the integrity and utility of the reference dataset:
1. Identify the key elements of the question.
2. Locate the specific section of the text that pertains to these elements.
3. Extract and present the relevant information, ensuring it directly addresses the question.
4. If necessary, rephrase or summarize the text to clarify the answer, but maintain the original meaning.

***
{context}
***
Question: {question}

Answer:
[Begin your response here, ensuring it is concise, relevant, and supported by the text.]
"""