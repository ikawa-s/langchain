import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

api_key = os.environ.get("GEMINI_API_KEY")

# Geminiモデルの初期化
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

# エージェントの作成
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# 実行
responce = agent.invoke(
    {"messages": [{"role": "user", "content": "東京の天気はなんですか？"}]}
)

print(responce)
