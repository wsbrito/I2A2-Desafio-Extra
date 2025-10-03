"""
I2A2 - Desafio Extra - Enhanced with Dynamic Plotting
Autor: Wagner dos Santos Brito (Enhanced by Claude)
Data: 02/10/2025
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()  # this will read .env and set environment variables


# -------------------------
# Configuration / Constants
# -------------------------
MEMORY_FILENAME = "memory.json"

# Enhanced system prompt for plotting
SYSTEM_PROMPT = """You are an expert data analyst assistant with visualization capabilities.

You have access to:
1. Dataset summary with statistics, correlations, and structure
2. A plotting tool that can create custom visualizations

When users ask questions:
- Analyze the data context provided
- Answer in Portuguese (Brazilian)
- Be concise but informative
- When visualization would help, use the plot_data tool to create charts
- Suggest which type of plot would be most appropriate

Available plot types:
- histogram: distribution of a numeric column
- scatter: relationship between two numeric columns
- box: boxplot for outlier detection
- bar: categorical data counts or aggregations
- line: time series or sequential data
- correlation_heatmap: correlation matrix
- pair_plot: multiple scatter plots

Always provide clear reasoning and actionable insights."""

# -------------------------
# Global state for passing dataframe to tools
# -------------------------
_CURRENT_DF = None
_CURRENT_SUMMARY = None

def set_current_data(df: pd.DataFrame, summary: Dict[str, Any]):
    global _CURRENT_DF, _CURRENT_SUMMARY
    _CURRENT_DF = df
    _CURRENT_SUMMARY = summary

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
    if hasattr(uploaded_file, "read"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")
    else:
        return pd.read_csv(uploaded_file)

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    summary['n_rows'], summary['n_cols'] = df.shape
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    summary['dtypes'] = dtypes
    summary['columns'] = list(df.columns)
    summary['head'] = df.head(5).to_dict(orient="records")
    summary['missing'] = df.isnull().sum().to_dict()
    
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        desc = numeric.describe().to_dict()
        medians = numeric.median().to_dict()
        summary['numeric_describe'] = desc
        summary['numeric_medians'] = medians
        
        # Top correlations
        corr = numeric.corr()
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

def detect_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.01) -> Dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return {'method': 'none', 'n_outliers': 0, 'outlier_indices': []}
    
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(numeric)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(Xs)
    outlier_idx = np.where(preds == -1)[0].tolist()
    
    return {
        'method': 'isolation_forest',
        'n_outliers': len(outlier_idx),
        'outlier_indices': outlier_idx,
        'outlier_samples': numeric.iloc[outlier_idx].head(5).to_dict(orient="records")
    }

# -------------------------
# DYNAMIC PLOTTING TOOL
# -------------------------
def create_plot(plot_type: str, x_col: str = None, y_col: str = None, 
                color_col: str = None, title: str = None) -> Optional[Any]:
    """
    Dynamically creates plots based on agent requests.
    
    Args:
        plot_type: Type of plot (histogram, scatter, box, bar, line, correlation_heatmap)
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column for color encoding
        title: Plot title
    
    Returns:
        Plotly figure object or None
    """
    global _CURRENT_DF
    
    if _CURRENT_DF is None:
        return None
    
    df = _CURRENT_DF
    
    try:
        if plot_type == "histogram":
            if not x_col or x_col not in df.columns:
                return None
            fig = px.histogram(df, x=x_col, marginal="box", 
                             title=title or f"Distribuição de {x_col}")
            return fig
        
        elif plot_type == "scatter":
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return None
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col if color_col in df.columns else None,
                           title=title or f"{y_col} vs {x_col}")
            return fig
        
        elif plot_type == "box":
            if not y_col or y_col not in df.columns:
                return None
            fig = px.box(df, y=y_col, x=x_col if x_col in df.columns else None,
                        title=title or f"Boxplot de {y_col}")
            return fig
        
        elif plot_type == "bar":
            if not x_col or x_col not in df.columns:
                return None
            # Aggregate if y_col provided, otherwise count
            if y_col and y_col in df.columns:
                agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(agg_df, x=x_col, y=y_col,
                           title=title or f"Média de {y_col} por {x_col}")
            else:
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count',
                           title=title or f"Contagem de {x_col}")
            return fig
        
        elif plot_type == "line":
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return None
            fig = px.line(df, x=x_col, y=y_col, color=color_col if color_col in df.columns else None,
                        title=title or f"{y_col} ao longo de {x_col}")
            return fig
        
        elif plot_type == "correlation_heatmap":
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                return None
            corr = numeric.corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                          title=title or "Mapa de Calor de Correlação")
            return fig
        
        else:
            return None
            
    except Exception as e:
        st.error(f"Erro ao criar gráfico: {e}")
        return None

def plot_data_tool(query: str) -> str:
    """
    Tool for the agent to create visualizations.
    
    Expected format: "plot_type|x_col|y_col|color_col|title"
    Example: "scatter|Time|Amount||Transações ao longo do tempo"
    """
    try:
        parts = query.split("|")
        plot_type = parts[0].strip() if len(parts) > 0 else ""
        x_col = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        y_col = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        color_col = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
        title = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
        
        fig = create_plot(plot_type, x_col, y_col, color_col, title)
        
        if fig is not None:
            # Store figure in session state for display
            if 'generated_plots' not in st.session_state:
                st.session_state.generated_plots = []
            st.session_state.generated_plots.append(fig)
            return f"✓ Gráfico criado: {plot_type} ({x_col}, {y_col}). O gráfico será exibido abaixo."
        else:
            return f"✗ Não foi possível criar o gráfico. Verifique os parâmetros."
    
    except Exception as e:
        return f"✗ Erro ao criar gráfico: {str(e)}"

def get_data_context_tool(query: str) -> str:
    """Tool to provide data context to the agent."""
    global _CURRENT_SUMMARY
    
    if _CURRENT_SUMMARY is None:
        return "Nenhum dado carregado."
    
    # Return a focused subset based on query keywords
    context = {
        "shape": {"rows": _CURRENT_SUMMARY['n_rows'], "cols": _CURRENT_SUMMARY['n_cols']},
        "columns": _CURRENT_SUMMARY['columns'],
        "numeric_columns": list(_CURRENT_SUMMARY.get('numeric_describe', {}).keys()),
        "top_correlations": _CURRENT_SUMMARY.get('top_correlations', [])[:5],
        "dtypes_sample": dict(list(_CURRENT_SUMMARY.get('dtypes', {}).items())[:10])
    }
    
    return json.dumps(context, indent=2, ensure_ascii=False)

# -------------------------
# LangChain Agent Setup
# -------------------------
def create_agent_executor(summary: Dict[str, Any]):
    """Creates a LangChain agent with tools for plotting and data access."""
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY não definida.")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Define tools
    tools = [
        Tool(
            name="plot_data",
            func=plot_data_tool,
            description="""Cria visualizações de dados. Formato: 'plot_type|x_col|y_col|color_col|title'
            
            Tipos disponíveis:
            - histogram: distribuição (precisa x_col)
            - scatter: relação entre variáveis (precisa x_col e y_col)
            - box: boxplot para outliers (precisa y_col)
            - bar: dados categóricos (precisa x_col, y_col opcional)
            - line: séries temporais (precisa x_col e y_col)
            - correlation_heatmap: matriz de correlação (sem parâmetros)
            
            Exemplo: "scatter|Time|Amount||Transações por tempo"
            """
        ),
        Tool(
            name="get_data_context",
            func=get_data_context_tool,
            description="Obtém informações sobre a estrutura do dataset, colunas, correlações e estatísticas."
        )
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="I2A2 - Agente E.D.A.", layout="wide")
    st.title("🤖 I2A2 - Agente E.D.A.")
    st.subheader("Autor: Wagner dos Santos Brito")

    # Initialize session state
    if 'generated_plots' not in st.session_state:
        st.session_state.generated_plots = []

    # Sidebar
    st.sidebar.header("⚙️ Controles")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv", "txt"])
    
    if st.sidebar.button("🗑️ Limpar memória"):
        clear_memory()
        st.session_state.generated_plots = []
        st.sidebar.success("Memória limpa!")

    # Load dataframe
    df = None
    if uploaded_file is not None:
        try:
            df = safe_read_csv(uploaded_file)
            st.sidebar.success(f"✓ {uploaded_file.name} ({df.shape[0]}x{df.shape[1]})")
        except Exception as e:
            st.sidebar.error(f"Erro: {e}")
            st.stop()
    else:
        st.info("📤 Faça upload de um arquivo CSV para começar.")
        st.stop()

    # Precompute summary
    with st.spinner("Analisando dados..."):
        summary = summarize_dataframe(df)
        outliers = detect_outliers_isolation_forest(df)
        
        # Set global data
        set_current_data(df, summary)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Linhas", summary['n_rows'])
    col2.metric("📋 Colunas", summary['n_cols'])
    col3.metric("🔢 Numéricas", len(summary.get('numeric_describe', {})))
    col4.metric("⚠️ Outliers", outliers['n_outliers'])

    # Data preview
    with st.expander("👁️ Ver dados (primeiras 100 linhas)"):
        st.dataframe(df.head(100))

    # Agent interaction
    st.subheader("💬 Converse com o Agente")
    
    user_question = st.text_input(
        "Sua pergunta:",
        placeholder="Ex: Mostre a distribuição de Amount",
        key="user_input"
    )
    
    col_ask, col_clear = st.columns([1, 4])
    with col_ask:
        ask_button = st.button("🚀 Perguntar", type="primary")
    with col_clear:
        if st.button("🧹 Limpar gráficos"):
            st.session_state.generated_plots = []
            st.rerun()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Configure OPENAI_API_KEY para usar o agente.")
        api_key = st.text_input("OPENAI_API_KEY:", type="password")
        if st.button("Salvar"):
            os.environ['OPENAI_API_KEY'] = api_key
            st.success("✓ Chave salva!")
            st.rerun()
        st.stop()

    # Process question
    if ask_button and user_question.strip():
        try:
            # Create agent
            agent_executor = create_agent_executor(summary)
            
            # Clear previous plots for this question
            st.session_state.generated_plots = []
            
            # Run agent
            with st.spinner("🤔 Pensando..."):
                result = agent_executor.invoke({"input": user_question})
            
            # Display response
            st.markdown("### 📝 Resposta:")
            st.write(result['output'])
            
            # Save to memory
            save_memory({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "question": user_question,
                "answer": result['output']
            })
            
        except Exception as e:
            st.error(f"❌ Erro: {e}")

    # Display generated plots
    if st.session_state.generated_plots:
        st.markdown("### 📊 Gráficos Gerados:")
        for idx, fig in enumerate(st.session_state.generated_plots):
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{idx}")

    # Memory
    with st.expander("📚 Histórico de Conversas"):
        mem = load_memory()
        if mem:
            for entry in reversed(mem[-10:]):
                st.markdown(f"**Q:** {entry['question']}")
                st.write(entry['answer'])
                st.markdown("---")
        else:
            st.write("Nenhuma conversa anterior.")

if __name__ == "__main__":
    main()
