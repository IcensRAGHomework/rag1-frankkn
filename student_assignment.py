import re
import json
from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


def generate_hw01(question):

    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke([message])
    
    examples = [
        {
            "input": "2024年台灣10月紀念日有哪些?", 
            "output": {
                "Result": [
                    {
                        "date": "2024-10-10",
                        "name": "國慶日"
                    },
                    {
                        "date": "2024-10-11",
                        "name": "重陽節"
                    }
                ]
            }
        }
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
    # print(few_shot_prompt.invoke({}).to_messages())

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",  """你是台灣內政部負責紀念日及節日實施辦法的專業人員,
                            請以JSON格式返回全部符合問題的結果,結果可能很多個,
                            並且去掉了不需要的標記（例如 ```json 和 ```)
                            格式必須包含\"Result\" 
                        """),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    chain = final_prompt | llm
    response_content = chain.invoke({"input": "2024年台灣10月紀念日有哪些"}).content

    # 使用 json.dumps() 來美化輸出
    formatted_json = json.dumps(response_content, ensure_ascii=False, indent=4)
    return formatted_json

def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

if __name__ == "__main__":
    question = "2024年台灣10月紀念日有哪些?"
    generate_hw01(question)
    
