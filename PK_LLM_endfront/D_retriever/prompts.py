from langchain.prompts import PromptTemplate
summary_prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="""
Objective:
* To understand and distill the provided text, extracting key information that specifically addresses the posed question.
* The provided text comes from a literature excerpt, where useful information may involve the background,\
    objectives, methods, and conclusions of the study.

Instructions:
* Thoroughly read and comprehend the text.
* Identify and extract information of the text that are most relevant to the specified question.
* Structure the information using bullet points or a concise paragraph to enhance readability.
* Emphasize the critical points that directly respond to or elucidate the posed question.

Exclusions:
* Exclude any information not directly pertinent to the question to maintain focus and relevance.

Output Format:
* Prepare the information to function effectively as input for another LLM, \
    facilitating subsequent analysis or response generation.

Clarity and Precision:
* Ensure the information is explicitly clear and precise to eliminate ambiguities, \
    focusing on usability for further language model processing.
    
"""
    'Reply "Not applicable" if text is irrelevant. '
    "At the end of your response, provide a score from 1-10 on a newline "
    "indicating relevance to question. Do not explain your score. "
    "\n\n"
    "{text}\n\n"
    "Question: {question}\n"
    "Relevant Information Summary:",
)



answer_prompt = PromptTemplate(
    input_variables=['text', 'question'],
    template="""
As a chemistry expert, here's the detailed process I follow to answer a question:
1. **Analyze the Question**: Carefully read the question to understand its key points and the type of information required. Identify relevant chemistry-related knowledge and consider the context of the question.
2. **Examine the Document Content**: Thoroughly review the provided document to find relevant facts, data, and evidence. Note critical details and ensure the information is accurate.
3. **Develop an Evidence-Based Response**: Based on the document content and extracted evidence, create a coherent, detailed answer grounded in chemical principles. Ensure the answer is logical and consider possible alternative explanations.
4. **Explain the Reasoning**: Describe the reasoning and logical process used to derive the answer, demonstrating expertise in the chemistry field. If the answer involves experiments or calculations, briefly explain the procedure.
5. **Address Insufficient Evidence**: If there isn't enough evidence in the document, clearly state that you cannot answer or explain the question. Avoid guessing, and suggest possible next steps, such as further research or data collection.

***
Question: {question}
***
Document Content:
{text}
***
**Notes**
- Ensure every step is evidence-based, avoiding unfounded assumptions.
- Use professional chemical terminology and ensure the answer is clear and understandable.
- If needed, refer to other relevant resources or consult with experts.
"""
)

response_prompt= PromptTemplate(
    input_variables=['text', 'question','step_back_context'],
    template="""
As a chemistry expert, here's the detailed process I follow to answer a question:
1. **Analyze the Question**: Carefully read the question to understand its key points and the type of information required. Identify relevant chemistry-related knowledge and consider the context of the question.
2. **Examine the Document Content**: Thoroughly review the provided document to find relevant facts, data, and evidence. Note critical details and ensure the information is accurate.
3. **Develop an Evidence-Based Response**: Based on the document content and extracted evidence, create a coherent, detailed answer grounded in chemical principles. Ensure the answer is logical and consider possible alternative explanations.
4. **Explain the Reasoning**: Describe the reasoning and logical process used to derive the answer, demonstrating expertise in the chemistry field. If the answer involves experiments or calculations, briefly explain the procedure.
5. **Address Insufficient Evidence**: If there isn't enough evidence in the document, clearly state that you cannot answer or explain the question. Avoid guessing, and suggest possible next steps, such as further research or data collection.

***
Question: {question}
***
Document Content:
{text}
{step_back_context}
***
**Notes**
- Ensure every step is evidence-based, avoiding unfounded assumptions.
- Use professional chemical terminology and ensure the answer is clear and understandable.
- If needed, refer to other relevant resources or consult with experts.
"""
)