import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import uuid

# ページ設定
st.set_page_config(page_title="チャットボット", page_icon="💬")
st.title("💬 対話型チャットボット")

# LangGraphアプリケーションの初期化
@st.cache_resource
def create_app():
    """LangGraphアプリケーションを作成"""
    # Geminiモデルの初期化
    api_key = os.environ.get("GEMINI_API_KEY")
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.7
    )
    
    # モデルを呼び出す関数
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}
    
    # グラフの定義
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    
    # メモリの追加
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# アプリケーションを取得
app = create_app()

# セッション初期化（thread_idで会話を管理）
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーメッセージを表示・保存
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # アシスタントの応答
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            # LangGraphの設定（thread_idで会話を識別）
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # メッセージをLangGraph形式で送信
            input_message = HumanMessage(content=prompt)
            
            # ストリーミングで応答を取得
            response_text = ""
            for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
                # 最後のメッセージ（AIの応答）を取得
                if event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, 'content'):
                        response_text = last_message.content
            
            st.markdown(response_text)
    
    # アシスタントメッセージを保存
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# サイドバーに設定
with st.sidebar:
    st.header("設定")
    
    # 会話履歴をクリア
    if st.button("新しい会話を始める"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    # デバッグ情報
    with st.expander("デバッグ情報"):
        st.write(f"Thread ID: {st.session_state.thread_id}")
        st.write(f"メッセージ数: {len(st.session_state.messages)}")
        st.json(st.session_state.messages)