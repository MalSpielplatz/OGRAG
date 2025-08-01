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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import rdflib
from collections import defaultdict
import json
import pandas as pd
import math

# Load environment variables from .env
load_dotenv()

# ===================== OGRAG Helper Functions =====================
def uri_to_label(uri):
    uri_str = str(uri)
    return uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]

def create_equivalence_map(g):
    equivalence_map = defaultdict(set)
    for s, o in g.subject_objects(predicate=rdflib.OWL.sameAs):
        s_label = uri_to_label(s)
        o_label = uri_to_label(o)
        equivalence_map[s_label].add(o_label)
        equivalence_map[o_label].add(s_label)
    return dict(equivalence_map)

def calculate_idf(hyperedges):
    num_documents = len(hyperedges)
    doc_freq = defaultdict(int)
    all_terms = set()
    for edge in hyperedges:
        edge_terms = set(edge['entities'])
        for rel, vals in edge['edge_info'].items():
            edge_terms.add(rel)
            edge_terms.update(vals)
        for term in edge_terms:
            doc_freq[term] += 1
            all_terms.add(term)
    idf_scores = {
        term: math.log(num_documents / (1 + doc_freq.get(term, 0)))
        for term in all_terms
    }
    return idf_scores, all_terms

def build_hyperedges_for_ograg(rdf_path):
    g = rdflib.Graph()
    g.parse(rdf_path, format='xml')
    equivalence_map = create_equivalence_map(g)
    hyperedges = []
    subj_map = defaultdict(lambda: defaultdict(list))
    for s, p, o in g:
        s_label = uri_to_label(s)
        p_label = uri_to_label(p)
        o_label = str(o) if isinstance(o, rdflib.Literal) else uri_to_label(o)
        subj_map[s_label][p_label].append(o_label)
    for subj, rels in subj_map.items():
        ent_set = set([subj])
        for plist in rels.values():
            ent_set.update(plist)
        hyperedges.append({
            "subject": subj,
            "edge_info": dict(rels),
            "entities": list(ent_set)
        })
    idf_scores, all_terms_for_match = calculate_idf(hyperedges)
    return hyperedges, all_terms_for_match, equivalence_map, idf_scores

def user_input_to_hypergraph(query, all_terms):
    query_lc = query.lower()
    user_terms = set()
    for term in all_terms:
        if term.lower() in query_lc:
            user_terms.add(term)
    return user_terms

def expand_query_terms(terms, eq_map):
    expanded = set(terms)
    for term in terms:
        expanded.update(eq_map.get(term, set()))
    return expanded

def match_hyperedges_tfidf(query, hyperedges, all_terms, eq_map, idf, top_k=3):
    initial_user_terms = user_input_to_hypergraph(query, all_terms)
    expanded_user_terms = expand_query_terms(initial_user_terms, eq_map)
    scored_edges = []
    for edge in hyperedges:
        score = 0
        edge_content = set(edge['entities'])
        for rel, vals in edge['edge_info'].items():
            edge_content.add(rel)
            edge_content.update(vals)
        actual_matched_terms = expanded_user_terms.intersection(edge_content)
        if actual_matched_terms:
            for term in actual_matched_terms:
                score += idf.get(term, 0)
            scored_edges.append((score, actual_matched_terms, edge))
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    top_k_edges = [
        (edge, matched_terms, score)
        for score, matched_terms, edge in scored_edges
    ][:top_k]
    return initial_user_terms, top_k_edges

def build_context_from_edges(matched_edges):
    blocks = []
    for edge, matched_terms, score in matched_edges:
        block = (
            f"Subject: {edge['subject']}\n"
            f"Matched terms: {', '.join(sorted(list(matched_terms)))}\n"
            f"Entities: {', '.join(edge['entities'])}\n"
            f"Edge info: {edge['edge_info']}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else "Tidak ada konteks relevan ditemukan."


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
3.  **Ontology-Grounded RAG (OGRAG)**: GPT-4o dibantu konteks dari pemrosesan struktur ontologi RDF berbasis TF-IDF symbolic.
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
        st.info("Metode: **Ontology-Grounded RAG (TF-IDF Hypergraph)**. Pengetahuan diambil dari struktur RDF (query expansion & matching TF-IDF).")
        @st.cache_resource
        def cache_hypergraph():
            return build_hyperedges_for_ograg(RDF_PATH)
        hyperedges, all_terms_for_match, equivalence_map, idf_scores = cache_hypergraph()
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        prompt_template_ograg = """
        Anda adalah asisten AI medis. Jawab hanya menggunakan informasi dari "Konteks" berikut.
        Jika informasi tidak ada, jawab dengan "Informasi tidak ditemukan."

        Konteks:
        {context}

        Pertanyaan:
        {question}

        Jawaban:
        """
        prompt = PromptTemplate.from_template(prompt_template_ograg)

        def chain_ograg_tfidf(user_question):
            user_hypergraph, top_k_edges = match_hyperedges_tfidf(
                user_question, hyperedges, all_terms_for_match, equivalence_map, idf_scores, top_k=3
            )
            context = build_context_from_edges(top_k_edges)
            prompt_input = prompt.format(context=context, question=user_question)
            answer_obj = llm.invoke(prompt_input)
            answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)
            # Uncomment ini jika ingin tampilkan debug info di main UI:
            # st.write("User Hypergraph:", user_hypergraph)
            # for idx, (e, m, s) in enumerate(top_k_edges, 1):
            #     st.write(f"Top-{idx} Subject: {e['subject']}, Score: {s:.2f}, Matched terms: {sorted(list(m))}")
            return answer

        return chain_ograg_tfidf

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
