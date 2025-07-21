import streamlit as st
import os
from dotenv import load_dotenv

# BENAR (Cara Import)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import rdflib
from collections import defaultdict
import json

# Load environment variables from .env
load_dotenv()

# Helper Functions untuk Ontology-Grounded RAG (tidak berubah)
def uri_to_label(uri):
    uri_str = str(uri)
    return uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]

def build_hypergraph_from_rdf(rdf_path):
    g = rdflib.Graph()
    g.parse(rdf_path, format='xml')
    hyperedge_dict = defaultdict(list)
    for s, p, o in g:
        s_l, p_l, o_l = uri_to_label(s), uri_to_label(p), uri_to_label(o)
        if p_l == 'type' and o_l == 'NamedIndividual':
            continue
        hyperedge_dict[s_l].append((p_l, o_l))
    hyperedges = {f"edge_{i}": dict(facts) for i, (subj, facts) in enumerate(hyperedge_dict.items())}
    for subj_label, edge_id in zip(hyperedge_dict.keys(), hyperedges.keys()):
        hyperedges[edge_id]['subjek'] = subj_label
    return hyperedges

def hyperedges_to_docs(hyperedges):
    documents = []
    for edge_id, facts in hyperedges.items():
        content = json.dumps(facts, ensure_ascii=False, indent=2)
        doc = Document(page_content=content, metadata={"source": facts.get("subjek", "N/A")})
        documents.append(doc)
    return documents

# Streamlit App
st.set_page_config(page_title="Studi Perbandingan RAG", layout="wide")
st.title("🔬 Studi Perbandingan: Baseline vs. RAG vs. OGRAG")
st.markdown("""
Aplikasi ini membandingkan tiga pendekatan dalam menjawab pertanyaan medis:
1.  **Model Dasar (Tanpa RAG)**: GPT-4o menjawab murni berdasarkan pengetahuannya sendiri.
2.  **RAG Standar**: GPT-4o dibantu konteks dari file RDF yang dipecah (*chunking*) sebagai teks biasa.
3.  **Ontology-Grounded RAG (OGRAG)**: GPT-4o dibantu konteks dari pemrosesan struktur ontologi RDF.
""")


# Sidebar Configuration
st.sidebar.header("⚙️ Konfigurasi")

model_choice = st.sidebar.selectbox("Pilih Model OpenAI:", ["gpt-4o", "gpt-4o-mini"])

# DIUBAH: Menambahkan opsi ketiga "Model Dasar (Tanpa RAG)"
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

# DIUBAH: Nama fungsi diubah menjadi lebih umum
@st.cache_resource
def setup_chain(method, model_name, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model=model_name, temperature=0)

    # BARU: Logika untuk metode Model Dasar (Tanpa RAG)
    if method == "Model Dasar (Tanpa RAG)":
        st.info("Metode: **Model Dasar (Tanpa RAG)**. LLM akan menjawab berdasarkan pengetahuannya sendiri.")
        prompt_template = "Anda adalah asisten AI medis. Jawab pertanyaan berikut dengan akurat: {question}"
        prompt = PromptTemplate.from_template(prompt_template)
        # Chain sederhana tanpa retriever
        chain = prompt | llm | StrOutputParser()
        return chain

    # Logika untuk dua metode RAG
    else:
        # Prompt khusus untuk RAG yang memaksa jawaban dari konteks
        prompt_template_rag = """
        Anda adalah asisten AI medis yang cermat dan akurat.
        Gunakan HANYA informasi dari 'Konteks' di bawah untuk menjawab 'Pertanyaan'.
        Jangan sekali-kali menggunakan pengetahuan eksternal Anda.

        Konteks:
        {context}

        Pertanyaan:
        {question}

        Aturan Jawaban:
        1. Jika informasi untuk menjawab pertanyaan ADA di dalam konteks, berikan jawaban langsung berdasarkan informasi tersebut.
        2. Jika informasi TIDAK ADA di dalam konteks, jawab HANYA dengan kalimat: "Informasi tidak ditemukan dalam basis data."
        """
        prompt = PromptTemplate.from_template(prompt_template_rag)
        rdf_path = "Ontology Alodog tanpa peringatan.rdf"
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        if method == "Ontology-Grounded RAG (OGRAG)":
            st.info("Metode: **Ontology-Grounded RAG**. Data diproses sebagai hypergraph dari RDF.")
            hyperedges = build_hypergraph_from_rdf(rdf_path)
            documents = hyperedges_to_docs(hyperedges)
        else: # RAG Standar
            st.info("Metode: **RAG Standar**. File RDF dibaca sebagai teks mentah dan dipecah (*chunking*).")
            with open(rdf_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = [Document(page_content=chunk, metadata={"source": rdf_path}) for chunk in text_splitter.split_text(raw_text)]

        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()} |
            prompt |
            llm |
            StrOutputParser()
        )
        return rag_chain

# Button to setup system
if st.sidebar.button("Siapkan Sistem"):
    if openai_api_key:
        with st.spinner(f"Menyiapkan sistem dengan metode **{method_choice}**..."):
            st.session_state['chain'] = setup_chain(method_choice, model_choice, openai_api_key)
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
            
            # DIUBAH: Cara memanggil chain disesuaikan dengan metodenya
            if method_choice == "Model Dasar (Tanpa RAG)":
                # Chain dasar memerlukan input berupa dictionary
                answer = chain.invoke({"question": user_question})
            else:
                # Chain RAG bisa menerima input string langsung
                answer = chain.invoke(user_question)

        st.subheader("📌 Jawaban AI:")
        st.markdown(f"*(Jawaban dihasilkan menggunakan metode: **{method_choice}**)*")
        st.markdown(answer)
    else:
        st.error("⚠️ Silakan siapkan sistem di sidebar terlebih dahulu!")