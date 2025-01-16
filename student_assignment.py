import re
import json
from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def extract_events(content, question):
    # 嘗試從問題中提取年份和月份
    match_question_date = re.search(r"(\d{4})年.*?(\d{1,2})月", question)

    if not match_question_date:
        print(f"Unable to extract date from question: {question}")
        return json.dumps({"Error": "無法從問題中提取日期資訊"}, ensure_ascii=False, indent=4)
    
    year, month = match_question_date.groups()
    month = month.zfill(2)  # 確保月份是兩位數格式

    # 正則表達式提取日期和節日名稱
    pattern = r"\*\*(.*?)\*\*.*?(\d{1,2})月(\d{1,2})日"
    events = []
    for match in re.finditer(pattern, content):
        name, event_month, day = match.groups()
        if int(event_month) == int(month):  # 確保是同一個月份
            events.append({
                "date": f"{year}-{event_month.zfill(2)}-{day.zfill(2)}",
                "name": name.strip()
            })
    
    # 如果沒有找到事件，回傳空結果
    if not events:
        print(f"No events found in content: {content}")
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

if __name__ == "__main__":
    question = "2024年台灣10月紀念日有哪些?"
    result = generate_hw01(question)
    print(result)
