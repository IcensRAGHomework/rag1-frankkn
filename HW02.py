import json
from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from langchain.tools import tool

@tool("get_calendar_events")
def get_calendar_events_tool(input: str):
    """
    Fetch calendar events for a specific year and month in Taiwan.
    Input should be in the format 'YYYY-MM'.
    """
    api_key = "6wBgACs2YWxPit5i4YdGNQ30GUybl5kL"
    country = "TW"
    
    year, month = map(int, input.split('-'))
    
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
from langchain.schema import SystemMessage

def generate_hw02(question):
    # System message that provides instructions
    system_message = SystemMessage(
        content="""你是一位台灣日曆專家，負責回答有關特定月份的台灣節日問題。
        當回答問題時，請務必返回以下格式的 JSON:
        {
            "Result": [
                {
                    "date": "YYYY-MM-DD",
                    "name": "節日名稱"
                },
                ...
            ]
        }
        僅列出該月份的相關紀念日，其他資訊請不要包含。
        """
    )

    # Few-Shot example setup
    examples = [
        {
            "input": "2024年台灣10月紀念日有哪些?",
            "output": {
                "Result": [
                    {"date": "2024-10-10", "name": "國慶日"},
                    {"date": "2024-10-11", "name": "重陽節"}
                ]
            }
        }
    ]
    
    # Few-shot prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        ),
        examples=examples,
    )

    formatted_few_shot_prompt = few_shot_prompt.format(input=question)

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

    response = agent.invoke(
        {"input": question, 
         "chat_history": [{"role": "system", "content": system_message.content},
                          {"role": "system", "content": formatted_few_shot_prompt},
                          {"role": "human", "content": question}]
        }
    )

    print(response['output'])

    return response['output']

if __name__ == "__main__":
    question = "2024年台灣10月紀念日有哪些?"
    generate_hw02(question)

    
