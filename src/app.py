import os
import streamlit as st
from dotenv import load_dotenv
import configparser

# Import your chain
from graphs.graph import PostgreSQLChain

APP_TITLE = "crmGPT - Interactive Chat"
APP_ICON = "🤖"

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the Streamlit upper-right chrome
    st.markdown(
        """
        <style>
        [data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Model selection
    models = {
        "OpenAI GPT-4o-mini": "gpt-4-1106-preview",
        "OpenAI GPT-3.5-turbo": "gpt-3.5-turbo",
    }

    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        with st.expander("Settings"):
            m = st.radio("LLM to use", options=models.keys())
            model = models[m]

    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    messages = st.session_state.conversation_history

    # Display initial message
    if len(messages) == 0:
        WELCOME = "Hello! I'm crmGPT, your AI assistant. How can I assist you today?"
        messages.append({"role": "assistant", "content": WELCOME})

    for message in messages[:]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])

    # User input at the bottom of the chat
    query = st.chat_input("Enter your query:")

    if query:
        # Immediately append the user's message to the conversation history and display it
        messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Run your chain logic to get the response
        with st.spinner("Processing..."):
            try:
                output, st.session_state.conversation_history = run_chain_sql(
                    query, model, st.session_state.conversation_history
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check the input or the model configuration.")
                st.error(f"Exception type: {type(e).__name__}")
                #output = "An error occurred while processing your request."


        # Immediately display the agent's response
        with st.chat_message("assistant"):
            st.write(output)




def run_chain_sql(query, model, conversation_history):
    chain_sql = PostgreSQLChain(model)

    chain_sql.build_graph()

    compiled_chain = chain_sql.compile_chain()

    # Limit conversation history to the last N messages (e.g., last 4 messages)
    limited_conversation_history = conversation_history[-4:]

    # Enter the chain with the updated conversation history
    output = chain_sql.enter_chain(query, 
                                   compiled_chain, 
                                   limited_conversation_history)
    conversation_history.append({"role": "assistant", "content": output})

    return output, conversation_history

if __name__ == "__main__":
    main()
