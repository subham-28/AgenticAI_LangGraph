from urllib import response

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
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# CONFIG={'configurable':{'thread_id': 'thread-1'}}

llm = ChatOpenRouter(
    model="openai/gpt-oss-20b:free",
    temperature=0
)

client = MultiServerMCPClient(
    {
        "arith": {
            "transport":"stdio",
            "command":"python",
            "args":["main.py"],
        },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


async def build_graph():

    tools= await client.get_tools()
    print("Tools loaded from MCP servers:", tools)
    # make llm aware of the tools
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        query=state['messages']
        response=await llm_with_tools.ainvoke(query)
        return {'messages': [response]}

    tool_node=ToolNode(tools) # creates a tool node
    
    # conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
    # checkpointer=SqliteSaver(conn=conn)
    
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)
    
    chatbot=graph.compile()
    return chatbot

async def main():
    chatbot=await build_graph()
    response=await chatbot.ainvoke(
                    {'messages': [HumanMessage(content="Add an expense of $500 for udemy course on 5th November")]})
    print(response['messages'][-1].content)


if __name__ == "__main__":
    asyncio.run(main())