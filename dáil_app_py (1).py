# app.py
# -*- coding: utf-8 -*-

import os
import json
import boto3
import chromadb                          # Vector DB
import openai                            # model for generation
import streamlit as st                  # Streamlit UI
from thefuzz import process             # for fuzzy speaker lookup
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain.chat_models import ChatOpenAI

# 1. Load your OpenAI key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# 2. Download & cache the ChromaDB vector store from S3
@st.cache_resource
def download_debate_db_from_s3(bucket_name, s3_prefix, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            rel = os.path.relpath(obj["Key"], s3_prefix)
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(bucket_name, obj["Key"], dst)
    return local_dir

debate_db_path = download_debate_db_from_s3(
    bucket_name="joebucketai",
    s3_prefix="debate_db/",
    local_dir="/tmp/debate_db/"
)
chroma_client = chromadb.PersistentClient(path=debate_db_path)
collection     = chroma_client.get_collection("oireachtas_debates")

# 3. Few-shot prompt setup
with open("dail_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}?"),
    ("ai",    "{answer}\n")
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)
full_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an Irish parliament chatbot. Users will ask "
        "about politicians' opinions on topics. You will summarise "
        "their positions using only provided quotes with URL/year citations."
    )),
    few_shot_prompt,
    ("human", "{question}")
])

gpt_nano = ChatOpenAI(
    model_name="gpt-4.1-nano",
    temperature=0.9,
    openai_api_key=openai.api_key
)
chain_nano = full_prompt | gpt_nano

# 4. Helper functions
def search_speaker_position(speaker_name, topic, num_results=5):
    results = collection.query(
        query_texts=[topic],
        n_results=num_results,
        where={"speaker": speaker_name},
        include=["metadatas"]
    )
    if not results["metadatas"][0]:
        return f"No speeches found for {speaker_name} on {topic}."
    out = f"### {speaker_name}'s Position on '{topic}':\n"
    for i, md in enumerate(results["metadatas"][0], 1):
        url  = md["url"]
        text = md["text"][:500].replace("\n", " ")
        out += f"\n**Quote {i} (source: {url}):** {text}...\n"
    return out

def speaker_fuzzy_lookup(speaker, speaker_list):
    return process.extractOne(speaker, speaker_list)[0]

def generate_answer(speaker_name, topic, list_of_speakers):
    sp = speaker_fuzzy_lookup(speaker_name, list_of_speakers)
    quotes = search_speaker_position(sp, topic)
    if quotes.startswith("No speeches found"):
        return quotes
    resp = chain_nano.invoke({
        "question": f"Summarise {sp}'s position on {topic} using these quotes: {quotes}",
        "answer": ""
    })
    return resp.content

# 5. Load speakers
with open("speakers.txt", "r", encoding="utf-8") as f:
    SPEAKERS = [s.strip() for s in f if s.strip()]

# 6. Streamlit UI
def main():
    st.title("Dáil Speeches Explorer 🇮🇪")
    speaker = st.text_input("Speaker name", placeholder="e.g. Micheál Martin")
    topic   = st.text_input("Topic",         placeholder="e.g. housing, Brexit…")

    if st.button("Search"):
        if not speaker or not topic:
            st.warning("Please enter both a speaker and a topic.")
        else:
            with st.spinner("🔍 Fetching quotes…"):
                answer = generate_answer(speaker, topic, SPEAKERS)
            st.markdown(answer)

    # Sidebar: Download requirements.txt
    st.sidebar.header("Resources")
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = f.read()
    st.sidebar.download_button(
        label="📥 Download requirements.txt",
        data=reqs,
        file_name="requirements.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
