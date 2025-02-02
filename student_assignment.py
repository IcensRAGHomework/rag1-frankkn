import json
from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.agents import Tool
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

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


    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",  """你是台灣內政部負責紀念日及節日實施辦法的專業人員,
                            請務必返回以下格式的 JSON
                            並且property name enclosed in double quotes:
                            {{format_instructions}}
                            並且去掉了不需要的標記（例如 ```json 和 ```)
                            僅列出該月份的相關紀念日，其他資訊請不要包含。
                        """),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    chain = final_prompt | llm
    response_content = chain.invoke({"input": "2024年台灣10月紀念日有哪些"}).content

    print(response_content)

    return response_content

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

def generate_hw02(question):

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

    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    # System message that provides instructions
    system_message = SystemMessage(
        content="""你是一位台灣日曆專家，負責回答有關特定月份的台灣節日問題。
        當回答問題時，請務必返回以下格式的 JSON:
        {{format_instructions}}
        並且去掉了不需要的標記（例如 ```json 和 ```)
        僅列出該月份的相關紀念日，其他資訊請不要包含。
        """
    )

    formatted_few_shot_messages = few_shot_prompt.format_messages()

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message.content),
            MessagesPlaceholder(variable_name="chat_history", optional=True),  
            *formatted_few_shot_messages, 
            ("human", "{input}"),  
            MessagesPlaceholder(variable_name="agent_scratchpad"),  
        ]
    )

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

    agent = create_tool_calling_agent(llm, tools, final_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke({"input": question})

    print(response['output'])

    return response['output']

def generate_hw03(question2, question3):
    
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

    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    # System message that provides instructions
    system_message = SystemMessage(
        content="""你是一位台灣日曆專家，負責回答有關特定月份的台灣節日問題。
        當回答有關某月份的節日問題時，請務必返回以下格式的 JSON:
        {{format_instructions}}
        僅列出該月份的相關紀念日，其他資訊請不要包含。

        當回答有關是否在先先前節日清單問題時,請務必返回以下格式的 JSON:
        其中
        Result的value是一個dictionary
        add表示是否需要將節日新增到節日清單中.根據問題判斷該節日是否存在於清單中,如果不存在,則為 true;否則為false.
        reason必須描述為什麼需要或不需要新增節日,具體說明是否該節日已經存在於清單中,以及當前清單的內容.
        {{format_instructions}}
        """
    )

    # few-shot examples  MessagesPlaceholder
    formatted_few_shot_messages = few_shot_prompt.format_messages()

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message.content),
            MessagesPlaceholder(variable_name="chat_history", optional=True),  
            *formatted_few_shot_messages,  
            ("human", "{input}"),  
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

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

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent = create_tool_calling_agent(llm, tools, final_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response1 = agent_with_chat_history.invoke(
        {"input": question2},
        config={"configurable": {"session_id": "<foo>"}},
    )

    response2 = agent_with_chat_history.invoke(
        {"input": question3},
        config={"configurable": {"session_id": "<foo>"}},
    )

    print(f"Responese2 = {response2['output']}")

    return response2['output']

    
def generate_hw04(question):

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    import base64
    from mimetypes import guess_type

    # Function to encode a local image into data URL 
    def local_image_to_data_url(image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    # Example usage
    image_path = './baseball.png'
    data_url = local_image_to_data_url(image_path)
    # print("Data URL:", data_url)

    from langchain_core.output_parsers import JsonOutputParser
    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    system_message = SystemMessage(
        content = 
            """ 
            請以JSON格式返回符合問題的結果
            並且去掉了不需要的標記（例如 ```json 和 ```)
            格式必須包含\"Result\": {"score": } 
            請務必返回以下格式的 JSON:
            {format_instructions}
            """
    )

    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                        "url": data_url
                    }}
            ]
    )

    response = llm.invoke([system_message, message])

    print(response.content)

    return response.content
    
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
    question2 = "2024年台灣10月紀念日有哪些?"
    question3 = "根據先前的節日清單，這個節日{\"date\": \"10-31\", \"name\": \"蔣公誕辰紀念日\"}是否有在該月份清單？"
    # generate_hw01(question2)
    # generate_hw02(question2)
    # generate_hw03(question2, question3)

    question = "請問中華台北的積分是多少"
    # generate_hw04(question)