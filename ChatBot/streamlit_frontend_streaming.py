import streamlit as st
from langgraph_backend import workflow
from langchain_core.messages import HumanMessage

st.title("🤖 AI Chat Assistant")
st.caption("Powered by LangGraph")



CONFIG={'configurable':{'thread_id': 'thread-1'}}

# session_state -> dict ->
 
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

# {'role':'user','content':'Hi'}
# {'role':'assistant','content':'Hello'}

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input("Type your message and press Enter...")

if user_input:
    # add to msg hist
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
       with st.spinner("Thinking..."):
            ai_message=st.write_stream(
                message_chunk.content for message_chunk,metadata in workflow.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config={'configurable':{'thread_id': 'thread-1'}},
                    stream_mode='messages'
                )
            )
       
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})


