import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

LLMFOUNDRY_TOKEN = os.getenv("LLMFOUNDRY_TOKEN")
BASE_URL = os.getenv("BASE_URL")

client = OpenAI(
    api_key=f"{LLMFOUNDRY_TOKEN}:pharmacy-bot",
    base_url=BASE_URL,
) 

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_pickle("medicine_embeddings.pkl")

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

if "context" not in st.session_state:
    st.session_state.context = [ {'role':'system', 'content':"""
You are PharmaBot, a friendly and efficient virtual assistant for an online pharmacy store. \

Your primary role is to assist customers in finding the right medications, healthcare products, and wellness items, \
answering their questions. \

For every user query (tagged as user_query), a relevant context (tagged as medical_context) from an internal medical \
database will be provided. Respond strictly based on this context unless the user query is has no medical context. \
If the provided context does not relate to the user's query, \
Explicitly state: "The internal medical database does not have sufficient information to answer your query." \
Then, offer general advice or suggestions for managing the condition, ensuring the information is helpful and accurate. \
                                  
If the query is too vague (e.g., "I have a skin allergy"), politely request more specific details, such as symptoms, \
duration, or triggers, to provide a meaningful response. \

If the user’s question is not related to medical needs or health concerns, politely inform them that you can only assist \
with medical or wellness-related queries and cannot provide information on unrelated topics. \

You provide detailed product information, including descriptions, indications, dosages, potential side effects, \
usage instructions, and any necessary precautions. You also clarify whether the customer requires a prescription for certain items. \
                                 
Respond in a warm and approachable manner, and ensure all interactions prioritize customer safety and accuracy. \
"""} ]
    
st.title("PharmaBot")
st.write("Hi. I'm PharmaBot, your medical AI assistant. How can I help you today?")
    
if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("",placeholder="Type your query here...")

if st.button("Send"):
    if user_input.strip():
        query_embedding = model.encode(user_input)

        df['similarity'] = df['usage_embeddings'].apply(lambda x: cosine_similarity([x], [query_embedding])[0][0])
        closest_match = df.sort_values(by='similarity', ascending=False).iloc[0]
        
        content =  "medical_context: " + str(closest_match) + ", user_query: " + user_input

        st.session_state.context.append({'role': 'user', 'content': content})
        
        response = get_completion_from_messages(st.session_state.context)
        st.session_state.context.append({'role': 'assistant', 'content': response})
        
        st.session_state.conversation.append((f"**You:** {user_input}", f"**PharmaBot:** {response}"))
    else:
        st.warning("Please enter a message before sending.")

if st.session_state.conversation:
    for user_msg, bot_msg in st.session_state.conversation:
        st.markdown(user_msg)
        st.markdown(bot_msg)