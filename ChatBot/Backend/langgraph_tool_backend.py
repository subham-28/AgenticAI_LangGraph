from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_openrouter import ChatOpenRouter
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random


load_dotenv()

CONFIG={'configurable':{'thread_id': 'thread-1'}}



llm = ChatOpenRouter(
    model="openai/gpt-oss-20b:free",
    temperature=0
)


# Tools

search_tool=DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    ALWAYS use this tool for any mathematical calculation.
    Use this tool whenever the user asks to add, subtract,
    multiply, divide, calculate percentages, or perform arithmetic.
    Never compute arithmetic yourself.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=O3BIQ9358W0US1KQ"
    r = requests.get(url)
    return r.json()


tools=[search_tool, calculator, get_stock_price]

# make llm aware of the tools
llm_with_tools = llm.bind_tools(tools)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state: ChatState):
    query=state['messages']
    response=llm_with_tools.invoke(query)
    return {'messages': [response]}

tool_node=ToolNode(tools) # creates a tool node

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)


checkpointer=SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

workflow=graph.compile(checkpointer=checkpointer)

# #test
# response=workflow.invoke(
#                     {'messages': [HumanMessage(content="Hi my name is Subham")]},
#                     config=CONFIG,
#                 )
# print(response)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
