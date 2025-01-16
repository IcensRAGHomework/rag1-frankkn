import re
import json
from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def extract_events(content, question):
    # 從問題中提取年份和月份
    match_question_date = re.search(r"(\d{4})年.*?(\d{1,2})月", question)
    if not match_question_date:
        return json.dumps({"Error": "無法從問題中提取日期資訊"}, ensure_ascii=False, indent=4)

    year, month = match_question_date.groups()
    month = month.zfill(2)  # 確保月份是兩位數格式

    # 更新正則表達式：匹配名稱在前或日期在前的情況
    pattern = r"\*\*(.*?)\*\*.*?(\d{1,2})月(\d{1,2})日|\*\*(\d{1,2})月(\d{1,2})日\s*-\s*(.+)\*\*"

    events = []
    for match in re.finditer(pattern, content):
        # 檢查兩種情況並提取結果
        if match.group(1):  # 名稱在前的情況
            name, event_month, day = match.group(1), match.group(2), match.group(3)
        elif match.group(4):  # 日期在前的情況
            event_month, day, name = match.group(4), match.group(5), match.group(6)
        else:
            continue

        # 確保月份匹配
        if int(event_month) == int(month):
            events.append({
                "date": f"{year}-{event_month.zfill(2)}-{day.zfill(2)}",
                "name": name.strip()
            })

    # 如果沒有找到事件，回傳空結果
    if not events:
        return json.dumps({"Result": []}, ensure_ascii=False, indent=4)

    # 格式化為 JSON 結果
    result = {"Result": events}
    return json.dumps(result, ensure_ascii=False, indent=4)

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

    content = response.content
    return extract_events(content, question)

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
    result = generate_hw01(question)
    print(result)
