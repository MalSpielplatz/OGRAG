import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
import rdflib
from collections import defaultdict
import pandas as pd

# Load environment variables from .env
load_dotenv()

# ===================== Helper Hypergraph & Vectorstore =====================

def uri_to_label(uri):
    uri_str = str(uri)
    return uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]

def construct_hypergraph_from_rdf(rdf_path):
    g = rdflib.Graph()
    g.parse(rdf_path, format='xml')
    subj_map = defaultdict(lambda: defaultdict(list))
    for s, p, o in g:
        s_label = uri_to_label(s)
        p_label = uri_to_label(p)
        o_label = str(o) if isinstance(o, rdflib.Literal) else uri_to_label(o)
        subj_map[s_label][p_label].append(o_label)
    hyperedges = []
    for i, (subj, rels) in enumerate(subj_map.items()):
        nodes = []
        for pred, objs in rels.items():
            for j, obj in enumerate(objs):
                nodes.append({"node_id": f"{i}-{j}", "text": f"{subj} {pred} {obj}", "parent_subject": subj})
        hyperedges.append({
            "edge_id": i,
            "subject": subj,
            "relations": dict(rels),
            "nodes": nodes
        })
    return hyperedges

def create_faiss_index(hyperedges, embedding_model):
    all_docs = []
    for edge in hyperedges:
        for node in edge['nodes']:
            doc = Document(
                page_content=node['text'],
                metadata={"subject": node['parent_subject']}
            )
            all_docs.append(doc)
    faiss_index = FAISS.from_documents(documents=all_docs, embedding=embedding_model)
    return faiss_index

def retrieve_context_from_faiss_with_scores(query, faiss_index, all_hyperedges, top_k=3):
    retrieved_docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
    retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
    relevant_subjects = list(dict.fromkeys([doc.metadata['subject'] for doc in retrieved_docs]))
    subject_to_hyperedge = {edge['subject']: edge for edge in all_hyperedges}
    matched_hyperedges = [subject_to_hyperedge[subj] for subj in relevant_subjects if subj in subject_to_hyperedge]
    context = build_context_from_hyperedges(matched_hyperedges)
    return context, retrieved_docs_with_scores

def build_context_from_hyperedges(matched_edges):
    if not matched_edges: return "Tidak ada konteks relevan yang ditemukan."
    context_str = ""
    for edge in matched_edges:
        context_str += f"Fakta terkait: {edge['subject']}\n"
        for relation, objects in edge['relations'].items():
            for obj in objects:
                context_str += f"- {relation}: {obj}\n"
        context_str += "\n"
    return context_str.strip()

# ===================== Helper for RAG (CSV) =====================
def load_csv_docs(csv_url):
    df = pd.read_csv(csv_url)
    def row_to_doc(row):
        chunks = []
        for col in df.columns:
            if pd.notnull(row[col]):
                chunks.append(f"{col.capitalize()}: {row[col]}")
        return ". ".join(chunks)
    documents = []
    for idx, row in df.iterrows():
        content = row_to_doc(row)
        doc = Document(page_content=content, metadata={"row": idx})
        documents.append(doc)
    return documents

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Studi Perbandingan RAG", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"] { width: 310px !important; }
        [data-testid="stSidebar"] > div:first-child { width: 310px !important; }
        .stSlider { padding-top: 10px; padding-bottom: 10px; }
        .stSlider > div[data-baseweb="slider"] > div { padding-left: 8px; padding-right: 8px; }
        .stSlider .css-1c9dki1 { background-color: #ff4b4b !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Pengembangan Chatbot dengan Retrieval Augmented Generation dan Integrasi Ontologi")
st.markdown("""
Aplikasi ini memiliki beberapa model yang bisa digunakan untuk menjawab pertanyaan medis:
1.  **Model Dasar (Tanpa RAG)**: GPT-4o menjawab murni berdasarkan pengetahuannya sendiri.
2.  **RAG Standar**: GPT-4o dibantu konteks dari CSV yang terstruktur sebagai basis pengetahuan.
3.  **Ontology-Grounded RAG (OGRAG)**: GPT-4o dibantu konteks dari pemrosesan struktur ontologi RDF berbasis vectorstore FAISS.
""")

st.sidebar.header("⚙️ Konfigurasi")

model_choice = st.sidebar.selectbox(" Pilih Model OpenAI:", ["gpt-4o", "gpt-4o-mini"])
temperature_value = st.sidebar.slider("Temperature (0 = deterministik, 1 = kreatif)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

method_choice = st.sidebar.selectbox(
    "Pilih Metode:",
    [
        "Model Dasar (Tanpa RAG)",
        "RAG Standar",
        "Ontology-Grounded RAG (OGRAG)"
    ]
)
openai_api_key = st.sidebar.text_input("Masukkan OpenAI API Key Anda", type="password")

if 'chain' not in st.session_state:
    st.session_state['chain'] = None

CSV_URL = "punya RAG.csv"  # <- Ganti dengan path file CSV kamu
RDF_PATH = "Ontology Alodog tanpa peringatan.rdf"  # <- Ganti dengan path file RDF kamu

@st.cache_resource
def setup_chain(method, model_name, api_key, temperature):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Pakai yang sama seperti pipeline utama!

    if method == "Model Dasar (Tanpa RAG)":
        st.info("Metode: **Model Dasar (Tanpa RAG)**. LLM akan menjawab berdasarkan pengetahuannya sendiri.")
        prompt_template = "Anda adalah asisten AI medis. Jawab pertanyaan berikut dengan akurat: {question}"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        return chain

    elif method == "RAG Standar":
        st.info("Metode: **RAG Standar**. Pengetahuan diambil dari CSV yang terstruktur.")
        documents = load_csv_docs(CSV_URL)
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        prompt_template_rag = """
Anda adalah asisten medis yang hanya boleh menggunakan informasi berikut (konteks) untuk menjawab pertanyaan. Jangan gunakan pengetahuan luar.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""
        prompt = PromptTemplate.from_template(prompt_template_rag)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()} |
            prompt |
            llm |
            StrOutputParser()
        )
        return rag_chain

    elif method == "Ontology-Grounded RAG (OGRAG)":
        st.info("Metode: **Ontology-Grounded RAG (FAISS Hypergraph)**. Pengetahuan diambil dari struktur RDF yang sudah diubah menjadi vectorstore dan hanya digunakan dari hasil retrieval.")
        @st.cache_resource
        def cache_hypergraph_and_faiss():
            hyperedges = construct_hypergraph_from_rdf(RDF_PATH)
            faiss_index = create_faiss_index(hyperedges, embeddings)
            return hyperedges, faiss_index
        hyperedges, faiss_index = cache_hypergraph_and_faiss()
        prompt_template_ograg = """
Anda adalah asisten AI medis yang akurat. Jawab pertanyaan hanya berdasarkan informasi dari "Konteks" yang disediakan.
Jika informasi tidak ada dalam konteks, jawab dengan "Informasi tidak ditemukan dalam konteks yang diberikan."

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""
        prompt = PromptTemplate.from_template(prompt_template_ograg)

        def chain_ograg_faiss(user_question):
            context, retrieved_docs_with_scores = retrieve_context_from_faiss_with_scores(
                user_question, faiss_index, hyperedges, top_k=3
            )
            prompt_input = prompt.format(context=context, question=user_question)
            answer_obj = llm.invoke(prompt_input)
            answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)
            # -- (Opsional, debug info) --
            # st.write("Top contexts:", context)
            # for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
            #     st.write(f"Top-{i}: Score: {score:.4f}, Content: {doc.page_content}")
            return answer

        return chain_ograg_faiss

# Tombol Siapkan Sistem
if st.sidebar.button("Siapkan Sistem"):
    if openai_api_key:
        with st.spinner(f"Menyiapkan sistem dengan metode **{method_choice}**..."):
            st.session_state['chain'] = setup_chain(method_choice, model_choice, openai_api_key, temperature_value)
        if st.session_state['chain']:
            st.sidebar.success("✅ Sistem siap!")
    else:
        st.sidebar.error("⚠️ Harap isi API key OpenAI!")

# User Input
st.header("💬 Tanya AI")
user_question = st.text_input("Masukkan pertanyaan Anda di sini:")

if st.button("Dapatkan Jawaban"):
    if st.session_state['chain']:
        with st.spinner("Mencari jawaban..."):
            chain = st.session_state['chain']
            if method_choice == "Model Dasar (Tanpa RAG)":
                answer = chain.invoke({"question": user_question})
            elif method_choice == "Ontology-Grounded RAG (OGRAG)":
                answer = chain(user_question)
            else:
                answer = chain.invoke(user_question)
        st.subheader("📌 Jawaban AI:")
        st.markdown(f"*(Jawaban dihasilkan menggunakan metode: **{method_choice}**, temperature: `{temperature_value}`)*")
        st.markdown(answer)
    else:
        st.error("⚠️ Silakan siapkan sistem di sidebar terlebih dahulu!")
