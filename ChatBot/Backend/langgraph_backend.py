from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_openrouter import ChatOpenRouter


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


checkpointer=InMemorySaver()

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

workflow=graph.compile(checkpointer=checkpointer)

workflow.invoke(
                    {'messages': [HumanMessage(content='Hi my name is Subham')]},
                    config=CONFIG
                )

print(workflow.get_state(config=CONFIG).values['messages'])