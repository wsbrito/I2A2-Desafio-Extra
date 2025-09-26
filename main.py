"""
I2A2 - Desafio Extra
Autor: Wagner dos Santos Brito
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# LangChain / OpenAI
#from langchain import OpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#from dotenv import load_dotenv
#load_dotenv()  # this will read .env and set environment variables


# -------------------------
# Configuration / Constants
# -------------------------
MEMORY_FILENAME = "memory.json"

#SYSTEM_PROMPT = """
#You are an expert data analyst assistant. You will be given:
#- A short structured dataset summary (columns, data types, sample rows)
#- Precomputed statistics (means, medians, standard deviations, top correlations, outliers summary)
#Your task: answer the user's question using only information given, explain reasoning simply,
#and produce a clear conclusion. When appropriate suggest a specific precomputed chart to display,
#naming it exactly (e.g., "histogram: Amount", "corr_heatmap", "boxplot: V12", "kmeans_clusters").
#If the question asks for reproducible code, generate Python code that uses pandas/plotly/sklearn and fits context.
#Be concise but informative. Always answer in Portuguese.
#"""

SYSTEM_PROMPT = """
You are a data analyst assistant. Input: dataset summary + stats.
Task: answer user in Portuguese, concise reasoning, clear conclusion.
If chart exists, name exactly (e.g., "histogram: Amount", "corr_heatmap").
If code requested, use pandas/plotly/sklearn.
"""


# -------------------------
# Utility: Memory
# -------------------------
def load_memory(path: str = MEMORY_FILENAME) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(entry: Dict[str, Any], path: str = MEMORY_FILENAME):
    mem = load_memory(path)
    mem.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

def clear_memory(path: str = MEMORY_FILENAME):
    if os.path.exists(path):
        os.remove(path)

# -------------------------
# EDA Helpers
# -------------------------
def safe_read_csv(uploaded_file) -> pd.DataFrame:
    # uploaded_file: UploadedFile or filepath
    if hasattr(uploaded_file, "read"):
        # Streamlit uploaded file
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            # try alternative separators
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")
    else:
        return pd.read_csv(uploaded_file)

def summarize_dataframe(df: pd.DataFrame, max_cols_show: int = 10) -> Dict[str, Any]:
    summary = {}
    summary['n_rows'], summary['n_cols'] = df.shape
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    summary['dtypes'] = dtypes
    summary['columns'] = list(df.columns)
    # top 5 sample rows
    summary['head'] = df.head(5).to_dict(orient="records")
    # counts of missing
    summary['missing'] = df.isnull().sum().to_dict()
    # basic numeric stats
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        desc = numeric.describe().to_dict()
        # compute medians separately
        medians = numeric.median().to_dict()
        summary['numeric_describe'] = desc
        summary['numeric_medians'] = medians
        # correlations
        corr = numeric.corr()
        # top absolute correlations
        corr_pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                a, b = cols[i], cols[j]
                corr_pairs.append((a, b, float(corr.iloc[i, j])))
        corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
        summary['top_correlations'] = corr_pairs_sorted
    else:
        summary['numeric_describe'] = {}
        summary['numeric_medians'] = {}
        summary['top_correlations'] = []
    return summary

def compute_histograms(df: pd.DataFrame, n_bins: int = 30, numeric_cols: List[str] = None) -> Dict[str, Any]:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plots = {}
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=n_bins, marginal="box", title=f"Histogram: {col}")
        plots[f"histogram: {col}"] = fig
    return plots

def compute_boxplots(df: pd.DataFrame, numeric_cols: List[str] = None) -> Dict[str, Any]:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plots = {}
    for col in numeric_cols:
        fig = px.box(df, y=col, points="outliers", title=f"Boxplot: {col}")
        plots[f"boxplot: {col}"] = fig
    return plots

def compute_corr_heatmap(df: pd.DataFrame) -> Any:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    # heatmap using plotly
    fig = px.imshow(corr, text_auto=True, title="Correlation heatmap")
    return fig

def detect_outliers_isolation_forest(df: pd.DataFrame, numeric_cols: List[str] = None, contamination: float = 0.01) -> Dict[str, Any]:
    result = {}
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        result['method'] = 'none'
        result['n_outliers'] = 0
        result['outlier_indices'] = []
        return result
    if numeric_cols:
        numeric = numeric[numeric_cols]
    # fill NA
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(numeric)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(Xs)  # -1 outlier, 1 inlier
    outlier_idx = np.where(preds == -1)[0].tolist()
    result['method'] = 'isolation_forest'
    result['n_outliers'] = len(outlier_idx)
    result['outlier_indices'] = outlier_idx
    # summary sample of outliers
    result['outlier_samples'] = numeric.iloc[outlier_idx].head(5).to_dict(orient="records")
    return result

def compute_kmeans(df: pd.DataFrame, numeric_cols: List[str] = None, n_clusters: int = 3) -> Dict[str, Any]:
    result = {}
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        result['method'] = 'none'
        result['n_clusters'] = 0
        return result
    if numeric_cols:
        numeric = numeric[numeric_cols]
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(numeric)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(Xs)
    result['method'] = 'kmeans'
    result['n_clusters'] = int(n_clusters)
    result['labels_sample_counts'] = dict(pd.Series(labels).value_counts().to_dict())
    # attach labels to a small sample for display
    sample_df = numeric.copy().reset_index(drop=True)
    sample_df['_cluster'] = labels
    result['sample_clusters'] = sample_df.head(10).to_dict(orient="records")
    # generate 2D scatter for first two numeric columns (if possible)
    if Xs.shape[1] >= 2:
        fig = px.scatter(x=Xs[:, 0], y=Xs[:, 1], color=labels.astype(str), title=f"KMeans clusters (k={n_clusters})")
        result['plot'] = fig
    else:
        result['plot'] = None
    return result

def truncate_summary(summary, max_chars=2000):
    text = json.dumps(summary, default=str, ensure_ascii=False, separators=(',', ':'))
    return text[:max_chars]

# -------------------------
# LLM Helper
# -------------------------
def build_llm(summary: Dict[str, Any]) -> Tuple[OpenAI, LLMChain]:
    # Build OpenAI / LangChain objects. API key must be in env var OPENAI_API_KEY.
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set. Export your key before running.")
    llm = OpenAI(temperature=0.2, max_tokens=800)
    # Template: include the structured summary as JSON text
    prompt_template = """{system}

Summary:
{summary_json}

User question:
{user_question}

Answer concisely with reasoning and a short conclusion."""
    prompt = PromptTemplate(
        input_variables=["system", "summary_json", "user_question"],
        template=prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return llm, chain

# -----------------------------------------
# -----------------------------------------
def recuperar_open_ai_key(st):

    #with st.popover("Informar OPENAI_API_KEY"):
    #    st.write("OPENAI_API_KEY:")
    #    key = st.text_input("key")            
    #    if st.button("Submit"):
    #        os.environ['OPENAI_API_KEY'] = key
    #        st.success("OPENAI_API_KEY informada!")
    
    st.write("Informar OPENAI_API_KEY:")
    key = st.text_input("OPENAI_API_KEY:")
    
    if st.button("Gravar OPENAI_API_KEY"):
        os.environ['OPENAI_API_KEY'] = key
        st.success("OPENAI_API_KEY informada!")
        st.rerun() # Rerun to update the main app after dialog closes


# -------------------------
# Streamlit App
# -------------------------
def main():

    st.set_page_config(page_title="I2A2 - Desafio Extra", layout="wide")
    st.title("I2A2 Desafio Extra - Agente de E.D.A.")
    st.markdown(
        "Realize o upload de um arquivo CSV e faça perguntas. O agente realiza E.D.A. (estatísticas descritivas, distribuições, correlações, agrupamentos, outliers) e usa um LLM para produzir explicações. A memória é armazenada localmente."
    )

    # Sidebar controls
    st.sidebar.header("Controles & Memoria")
    uploaded_file = st.sidebar.file_uploader("Upload do arquivo CSV", type=["csv", "txt"])
    if st.sidebar.button("Limpar memoria do agente"):
        clear_memory()
        st.sidebar.success("Memoria limpa.")

    memory = load_memory()

    # Load dataframe
    df = None
    if uploaded_file is not None:
        try:
            df = safe_read_csv(uploaded_file)
            st.sidebar.success(f"Carregou {uploaded_file.name} ({df.shape[0]} linhas, {df.shape[1]} colunas)")
        except Exception as e:
            st.sidebar.error(f"Não foi possível ler o arquivo: {e}")
            st.stop()
    else:
        st.sidebar.write("Faça upload de um CSV para começar ou marque 'Usar amostra' para carregar uma amostra local.")

    if df is None:
        st.info("Aguardando o upload do CSV ou opte por usar uma amostra local. Após o carregamento, as análises pré-computadas aparecerão aqui.")
        st.stop()

    # Basic info & option to show raw data
    st.subheader("Visualização de dados")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Formato: {df.shape[0]} linha x {df.shape[1]} colunas")
        if st.checkbox("Mostrar dados brutos (primeiras 200 linhas)", value=False):
            st.dataframe(df.head(200))
    with col2:
        if st.checkbox("Mostrar tipos de colunas e contagens ausentes", value=False):
            dtypes = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isnull().sum()})
            st.table(dtypes)

    # Precompute summaries & plots (cached)
    @st.cache_data(show_spinner=False)
    def precompute(df: pd.DataFrame):
        summary = summarize_dataframe(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # limit numeric cols for heavy operations if too many
        if len(numeric_cols) > 30:
            numeric_for_plots = numeric_cols[:30]
        else:
            numeric_for_plots = numeric_cols
        histos = compute_histograms(df, numeric_cols=numeric_for_plots)
        boxes = compute_boxplots(df, numeric_cols=numeric_for_plots)
        corr = compute_corr_heatmap(df)
        outliers = detect_outliers_isolation_forest(df, numeric_cols=numeric_for_plots)
        kmeans = compute_kmeans(df, numeric_cols=numeric_for_plots, n_clusters=3)
        return {
            "summary": summary,
            "histograms": histos,
            "boxplots": boxes,
            "corr_heatmap": corr,
            "outliers": outliers,
            "kmeans": kmeans,
            "numeric_cols": numeric_for_plots
        }

    pre = precompute(df)
    summary = pre["summary"]

    # Show some quick summary cards
    st.subheader("Resumo rápido de EDA")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Linha", summary['n_rows'])
    col_b.metric("Colunas", summary['n_cols'])
    col_c.metric("Colunas numéricas", len(pre['numeric_cols']))
    col_d.metric("Outliers detectados (amostra)", pre['outliers']['n_outliers'])

    # Show top correlations
    st.markdown("**Principais correlações (absolutas)**")
    if summary['top_correlations']:
        corr_table = pd.DataFrame(summary['top_correlations'], columns=["var1", "var2", "corr"])
        st.dataframe(corr_table)
    else:
        st.write("Nenhuma correlação numérica disponível.")

    # Plots area: allow user to pick any precomputed plot to view
    st.subheader("Gráficos pré-computados")
    available_plots = list(pre['histograms'].keys()) + list(pre['boxplots'].keys())
    if pre['corr_heatmap'] is not None:
        available_plots.append("corr_heatmap")
    if pre['kmeans'].get('plot') is not None:
        available_plots.append("kmeans_clusters")  # alias name used by the LLM
    plot_choice = st.selectbox("Escolha o gráfico a ser exibido", options=["(none)"] + available_plots)
    if plot_choice != "(none)":
        if plot_choice.startswith("histogram:"):
            st.plotly_chart(pre['histograms'][plot_choice], use_container_width=True)
        elif plot_choice.startswith("boxplot:"):
            st.plotly_chart(pre['boxplots'][plot_choice], use_container_width=True)
        elif plot_choice == "corr_heatmap":
            st.plotly_chart(pre['corr_heatmap'], use_container_width=True)
        elif plot_choice == "kmeans_clusters":
            st.plotly_chart(pre['kmeans']['plot'], use_container_width=True)
        else:
            st.write("Gráfico não encontrato.")

    # Show outlier sample
    if st.checkbox("Mostrar amostra e índices de outliers", value=False):
        st.write(f"Método de detecção de outliers: {pre['outliers']['method']}")
        st.write(f"Número de outliers detectados: {pre['outliers']['n_outliers']}")
        if pre['outliers']['n_outliers'] > 0:
            st.write(pd.DataFrame(pre['outliers']['outlier_samples']))

    # ---- Conversational agent section ----
    st.subheader("Pergunte ao agente")
    st.markdown("Digite uma pergunta sobre o conjunto de dados (exemplo: 'Quais variáveis estão mais correlacionadas com o valor?' ou 'Existem padrões temporais?' ou 'Mostre-me a distribuição do valor da transação').")

    user_question = st.text_input("Sua pergunta", value="", key="user_question_input")
    ask_button = st.button("Pergunte ao agente")

    # Initialize LangChain LLM/chain
    if os.getenv("OPENAI_API_KEY") is None:
        recuperar_open_ai_key(st)
        st.warning("A chave da API OpenAI não está definida. Informe a OPENAI_API_KEY para habilitar o LLM. Você ainda pode visualizar gráficos/tabelas computados.")
    else:
        llm, chain = build_llm(summary)

    if ask_button and user_question.strip() != "":
        # Build compact structured summary JSON to send to LLM
        compact_summary = {
            "shape": {"rows": summary['n_rows'], "cols": summary['n_cols']},
            "dtypes": summary['dtypes'],
            "missing_counts_sample": dict(list(summary['missing'].items())[:10]),
            "numeric_describe_sample": {k: {stat: v for stat, v in list(stats.items())[:5]} for k, stats in summary.get('numeric_describe', {}).items()} if summary.get('numeric_describe') else {},
            "numeric_medians_sample": {k: v for k, v in list(summary.get('numeric_medians', {}).items())[:10]},
            "top_correlations": summary.get('top_correlations', []),
            "outliers": {"method": pre['outliers']['method'], "n_outliers": pre['outliers']['n_outliers']},
            "available_plots": ["histogram: <col>", "boxplot: <col>", "corr_heatmap", "kmeans_clusters"]
        }
        # Send to LLM
        if os.getenv("OPENAI_API_KEY") is None:
            st.error("Impossibilidade de executar a LLM: OPENAI_API_KEY não foi informada.")
        else:
            try:
                prompt_vars = {
                    "system": SYSTEM_PROMPT,
                    "summary_json": truncate_summary(compact_summary),
                    "user_question": user_question
                }
                with st.spinner("Processando..."):
                    response = chain.run(prompt_vars)
                # Save to memory
                mem_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "question": user_question,
                    "answer": response
                }
                save_memory(mem_entry)
                st.markdown("**Resposta do agente:**")
                st.write(response)
                st.markdown("---")
                st.info("Se a resposta do agente fizer referência a um nome de gráfico (por exemplo, 'histograma: Valor'), selecione-o acima para visualizar o gráfico.")
            except Exception as e:
                st.error(f"LLM call failed: {e}")

    # Memory viewer
    st.subheader("Memória do Agente (perguntas e respostas anteriores)")
    mem = load_memory()
    if mem:
        # Show the last 10 entries
        for entry in mem[-10:][::-1]:
            st.markdown(f"**Q ({entry['timestamp']}):** {entry['question']}")
            st.write(entry['answer'])
            st.markdown("---")
    else:
        st.write("A memória está vazia. Faça perguntas para preenchê-la.")

    # Footer: help & export
    st.markdown("----")
    st.markdown("**Exportar**: você pode baixar o resumo do conjunto de dados atual como JSON ou a memória como JSON.")
    if st.button("Baixar resumo JSON"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(tmp.name, "w", encoding="utf-8") as f:
            json.dump(pre['summary'], f, indent=2, ensure_ascii=False)
        st.success("Resumo salvo em arquivo. Baixe no link abaixo.")
        st.markdown(f"[Baixar arquivo de resumo]({tmp.name})")
    if st.button("Baixar memória JSON"):
        memfile = MEMORY_FILENAME if os.path.exists(MEMORY_FILENAME) else None
        if memfile:
            st.markdown(f"[Baixar arquivo de memória]({memfile})")
        else:
            st.write("Nenhum arquivo de memória presente.")

if __name__ == "__main__":
    main()

