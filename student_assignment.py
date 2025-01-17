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
    return response_content

from langchain.tools import tool

@tool("get_calendar_events")
def get_calendar_events_tool(input: str):
    """
    Fetch calendar events for a specific year and month in Taiwan.
    Input should be in the format 'YYYY-MM'.
    """
    api_key = "6wBgACs2YWxPit5i4YdGNQ30GUybl5kL"
    country = "TW"
    
    # 解析輸入年份與月份
    year, month = map(int, input.split('-'))
    
    # 呼叫 API
    result = call_calendarific_api(api_key, country, year, month)
    return json.dumps(result)

def call_calendarific_api(api_key, country, year, month):
    url = f'https://calendarific.com/api/v2/holidays?api_key={api_key}&country={country}&year={year}&month={month}'
    import requests
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"API request fail, Status:{response.status_code}")
    
    holidays = response.json().get('response', {}).get('holidays', [])
    
    result = {"Result": [{"date": holiday['date']['iso'], "name": holiday['name']} for holiday in holidays]}
    return result

from langchain.agents import initialize_agent, Tool
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage

# Initialize LLM
llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)

# Define tool
tools = [
    Tool(
        name="get_calendar_events",
        func=get_calendar_events_tool,
        description="Call this tool to fetch Taiwan's calendar events for a specific year and month in 'YYYY-MM' format."
    )
]

# Initialize AgentExecutor
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=None  # If need context(上下文記憶), use ConversationBufferMemory
)

def generate_hw02(question):

    system_message = SystemMessage(
        content="""你是一位台灣日曆專家，負責回答有關特定月份的台灣節日問題。
        如果問題提到具體年份和月份，請呼叫相關工具來獲取數據。
        """
    )

    response = agent.invoke(
        {"input": question, 
         "chat_history": [{"role": "system", "content": system_message.content}]
        }
    )
    
    print(response)

    return response

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
    # generate_hw01(question)
    generate_hw02(question)

    
