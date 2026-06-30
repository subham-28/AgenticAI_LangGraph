from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_openrouter import ChatOpenRouter
import sqlite3


load_dotenv()

CONFIG={'configurable':{'thread_id': 'thread-1'}}



llm = ChatOpenRouter(
    model="google/gemma-4-26b-a4b-it:free",
    temperature=0
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state: ChatState):
    query=state['messages']
    response=llm.invoke(query)
    return {'messages': [response]}

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)


checkpointer=SqliteSaver(conn=conn)

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

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
