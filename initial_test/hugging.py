from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import os


# HuggingFace Repository ID
repo_id = 'mistralai/Mistral-7B-v0.1'

# 질의내용
question = "Who is Son Heung?"

# 템플릿
template = """Question: {question}

Answer: """

# 프롬프트 템플릿 생성
prompt = PromptTemplate(template=template, input_variables=["question"])

# HuggingFaceHub 객체 생성
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.2,
                  "max_length": 512}
)

# LLM Chain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실행
print(llm_chain.run(question=question))