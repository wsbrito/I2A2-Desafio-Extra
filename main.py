"""
I2A2 - Agente EDA com Google Gemini (Gratuito)
Autor: Wagner dos Santos Brito (Enhanced)
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# LangChain com Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# this will read .env and set environment variables
from dotenv import load_dotenv
load_dotenv()  

# -------------------------
# Configuration
# -------------------------
MEMORY_FILENAME = "memory.json"

SYSTEM_PROMPT = """Você é um assistente especialista em análise de dados com capacidade de criar visualizações.

Você tem acesso a:
1. Contexto dos dados (get_data_context): estrutura, estatísticas, correlações
2. Ferramenta de plotagem (plot_data): para criar gráficos

IMPORTANTE: Sempre responda em Português do Brasil.

REGRA FUNDAMENTAL SOBRE GRÁFICOS:
- Crie gráficos SOMENTE quando o usuário solicitar explicitamente (palavras como: "mostre", "plote", "crie gráfico", "visualize", "exiba", "gere gráfico")
- Se o usuário apenas perguntar ou pedir análise SEM mencionar visualização, responda apenas com texto
- Não sugira criar gráficos, apenas crie quando explicitamente solicitado

Para criar gráficos quando solicitado, use este formato EXATO:
Action: plot_data
Action Input: tipo|coluna_x|coluna_y|coluna_cor|título

Tipos de gráfico disponíveis:
- histogram: distribuição de valores (precisa coluna_x)
- scatter: relação entre variáveis (precisa coluna_x e coluna_y)
- box: detecção de outliers (precisa coluna_y)
- bar: dados categóricos ou agregações (precisa coluna_x)
- line: séries temporais (precisa coluna_x e coluna_y)
- correlation_heatmap: matriz de correlação completa (sem parâmetros)

Exemplos de uso:
- histogram|Amount|||Distribuição de Valores
- scatter|Time|Amount||Valores ao longo do tempo
- correlation_heatmap||||Correlações

Exemplos de perguntas:
- "Qual a média de Amount?" → Responda SÓ com texto, SEM gráfico
- "Mostre a distribuição de Amount" → Responda com texto E crie histogram
- "Existe correlação entre X e Y?" → Responda SÓ com texto, SEM gráfico
- "Plote X versus Y" → Responda com texto E crie scatter plot

Seja direto, conciso e forneça insights valiosos sobre os dados."""

# -------------------------
# Global state
# -------------------------
_CURRENT_DF = None
_CURRENT_SUMMARY = None

def set_current_data(df: pd.DataFrame, summary: Dict[str, Any]):
    global _CURRENT_DF, _CURRENT_SUMMARY
    _CURRENT_DF = df
    _CURRENT_SUMMARY = summary

# -------------------------
# Memory Functions
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
# Data Processing
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
    summary['columns'] = list(df.columns)
    summary['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    summary['missing'] = {col: int(count) for col, count in df.isnull().sum().items() if count > 0}
    
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        desc = numeric.describe().to_dict()
        summary['numeric_stats'] = {
            col: {
                'mean': round(float(desc[col]['mean']), 2),
                'std': round(float(desc[col]['std']), 2),
                'min': round(float(desc[col]['min']), 2),
                'max': round(float(desc[col]['max']), 2)
            } for col in desc.keys()
        }
        
        # Top correlations
        corr = numeric.corr()
        corr_pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                a, b = cols[i], cols[j]
                val = float(corr.iloc[i, j])
                if not np.isnan(val):
                    corr_pairs.append((a, b, round(val, 3)))
        corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
        summary['top_correlations'] = corr_pairs_sorted
    else:
        summary['numeric_stats'] = {}
        summary['top_correlations'] = []
    
    return summary

def detect_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return {'n_outliers': 0, 'outlier_indices': []}
    
    try:
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(numeric)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(contamination=0.01, random_state=42)
        preds = iso.fit_predict(Xs)
        outlier_idx = np.where(preds == -1)[0].tolist()
        
        return {
            'n_outliers': len(outlier_idx),
            'outlier_indices': outlier_idx[:10]  # Primeiros 10
        }
    except:
        return {'n_outliers': 0, 'outlier_indices': []}

# -------------------------
# Plotting Functions
# -------------------------
def create_plot(plot_type: str, x_col: str = None, y_col: str = None, 
                color_col: str = None, title: str = None) -> Optional[Any]:
    global _CURRENT_DF
    
    if _CURRENT_DF is None:
        return None
    
    df = _CURRENT_DF
    
    try:
        if plot_type == "histogram":
            if not x_col or x_col not in df.columns:
                return None
            fig = px.histogram(df, x=x_col, marginal="box", 
                             title=title or f"Distribuição de {x_col}",
                             labels={x_col: x_col})
            fig.update_layout(showlegend=False)
            return fig
        
        elif plot_type == "scatter":
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return None
            fig = px.scatter(df, x=x_col, y=y_col, 
                           color=color_col if color_col and color_col in df.columns else None,
                           title=title or f"{y_col} vs {x_col}",
                           labels={x_col: x_col, y_col: y_col})
            return fig
        
        elif plot_type == "box":
            if not y_col or y_col not in df.columns:
                return None
            fig = px.box(df, y=y_col, 
                        x=x_col if x_col and x_col in df.columns else None,
                        title=title or f"Boxplot de {y_col}",
                        labels={y_col: y_col})
            return fig
        
        elif plot_type == "bar":
            if not x_col or x_col not in df.columns:
                return None
            if y_col and y_col in df.columns:
                agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                agg_df = agg_df.sort_values(y_col, ascending=False).head(20)
                fig = px.bar(agg_df, x=x_col, y=y_col,
                           title=title or f"Média de {y_col} por {x_col}",
                           labels={x_col: x_col, y_col: y_col})
            else:
                counts = df[x_col].value_counts().head(20).reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count',
                           title=title or f"Contagem de {x_col}",
                           labels={x_col: x_col, 'count': 'Contagem'})
            return fig
        
        elif plot_type == "line":
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return None
            df_sorted = df.sort_values(x_col)
            fig = px.line(df_sorted, x=x_col, y=y_col, 
                        color=color_col if color_col and color_col in df.columns else None,
                        title=title or f"{y_col} ao longo de {x_col}",
                        labels={x_col: x_col, y_col: y_col})
            return fig
        
        elif plot_type == "correlation_heatmap":
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                return None
            corr = numeric.corr()
            fig = px.imshow(corr, 
                          text_auto=".2f", 
                          aspect="auto",
                          color_continuous_scale="RdBu_r",
                          title=title or "Mapa de Calor de Correlação")
            fig.update_xaxes(side="bottom")
            return fig
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico: {e}")
        return None

def plot_data_tool(query: str) -> str:
    """Ferramenta para criar visualizações."""
    try:
        parts = [p.strip() for p in query.split("|")]
        plot_type = parts[0] if len(parts) > 0 else ""
        x_col = parts[1] if len(parts) > 1 and parts[1] else None
        y_col = parts[2] if len(parts) > 2 and parts[2] else None
        color_col = parts[3] if len(parts) > 3 and parts[3] else None
        title = parts[4] if len(parts) > 4 and parts[4] else None
        
        fig = create_plot(plot_type, x_col, y_col, color_col, title)
        
        if fig is not None:
            if 'generated_plots' not in st.session_state:
                st.session_state.generated_plots = []
            st.session_state.generated_plots.append(fig)
            return f"✓ Gráfico {plot_type} criado com sucesso! O gráfico será exibido abaixo da resposta."
        else:
            return f"✗ Não foi possível criar o gráfico {plot_type}. Verifique se as colunas '{x_col}' e '{y_col}' existem no dataset."
    
    except Exception as e:
        return f"✗ Erro ao processar: {str(e)}"

def get_data_context_tool(query: str) -> str:
    """Fornece informações sobre o dataset."""
    global _CURRENT_SUMMARY
    
    if _CURRENT_SUMMARY is None:
        return "Nenhum dado foi carregado ainda."
    
    context = {
        "total_linhas": _CURRENT_SUMMARY['n_rows'],
        "total_colunas": _CURRENT_SUMMARY['n_cols'],
        "colunas_disponiveis": _CURRENT_SUMMARY['columns'][:30],
        "colunas_numericas": list(_CURRENT_SUMMARY.get('numeric_stats', {}).keys()),
        "estatisticas_amostra": dict(list(_CURRENT_SUMMARY.get('numeric_stats', {}).items())[:5]),
        "top_5_correlacoes": _CURRENT_SUMMARY.get('top_correlations', [])[:5],
        "colunas_com_dados_faltantes": _CURRENT_SUMMARY.get('missing', {})
    }
    
    return json.dumps(context, indent=2, ensure_ascii=False)

# -------------------------
# Agent Setup with Gemini
# -------------------------
def create_agent_executor(google_api_key: str):
    """Cria o agente usando Google Gemini."""
    
    # Configurar Gemini - usando modelo correto
    llm = ChatGoogleGenerativeAI(
        model="gemma-3n-e2b-it",  # Modelo estável e compatível
        google_api_key=google_api_key,
        temperature=0.1,
        convert_system_message_to_human=True
    )
    
    # Definir ferramentas
    tools = [
        Tool(
            name="plot_data",
            func=plot_data_tool,
            description="""Cria visualizações de dados. 
            Formato: 'tipo|coluna_x|coluna_y|coluna_cor|titulo'
            
            Tipos disponíveis:
            - histogram: para distribuições (precisa coluna_x)
            - scatter: para relações (precisa coluna_x e coluna_y)
            - box: para outliers (precisa coluna_y)
            - bar: para categorias (precisa coluna_x)
            - line: para séries temporais (precisa coluna_x e coluna_y)
            - correlation_heatmap: matriz de correlação (sem parâmetros)
            
            Exemplo: "histogram|Amount|||Distribuição de Valores"
            """
        ),
        Tool(
            name="get_data_context",
            func=get_data_context_tool,
            description="Obtém informações sobre a estrutura do dataset, colunas disponíveis, estatísticas e correlações."
        )
    ]
    
    # Template ReAct em português
    template = """Responda SEMPRE em Português do Brasil. 

Você tem acesso às seguintes ferramentas:

{tools}

Use EXATAMENTE este formato:

Question: a pergunta do usuário
Thought: você deve pensar sobre o que fazer
Action: a ação a tomar, deve ser uma de [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (este ciclo Thought/Action/Action Input/Observation pode repetir N vezes)
Thought: Agora eu sei a resposta final
Final Answer: a resposta final em português brasileiro

IMPORTANTE:
- Sempre use "Action:" e "Action Input:" exatamente como mostrado
- Para gráficos, use o formato: tipo|col_x|col_y|cor|titulo
- Seja conciso e direto nas respostas

Comece agora!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # Criar agente
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        early_stopping_method="generate"
    )
    
    return agent_executor

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="I2A2 - Desafio Extra - Agente E.D.A", layout="wide")
    st.title("🤖 I2A2 - Desafio Extra - Agente E.D.A")
    
    st.markdown("""
    ✨ **Autor: Wagner dos Santos Brito**
    
    📝 **Como obter sua chave API:**
    1. Acesse: https://aistudio.google.com/app/apikey
    2. Clique em "Create API Key" ou "Get API Key"
    3. Cole a chave na barra lateral
    """)

    api_key = os.environ["GOOGLE_API_KEY"]
    # Initialize session state
    if 'generated_plots' not in st.session_state:
        st.session_state.generated_plots = []
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")

    # Sidebar
    st.sidebar.header("⚙️ Configuração")
    
    
    if api_key:
        st.session_state.google_api_key = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        st.sidebar.success("✓ API Key configurada!")
    else:
        st.sidebar.warning("⚠️ Configure a API Key para usar o agente")
        # API Key input
        api_key = st.sidebar.text_input(
            "🔑 Google API Key:",
            value=st.session_state.google_api_key,
            type="password",
            help="Obtenha em: https://aistudio.google.com/app/apikey"
        )
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("📤 Upload CSV", type=["csv", "txt"])
    
    # Clear buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Limpar memória"):
            clear_memory()
            st.sidebar.success("Limpo!")
    with col2:
        if st.button("🧹 Limpar gráficos"):
            st.session_state.generated_plots = []
            st.rerun()

    # Load dataframe
    df = None
    if uploaded_file is not None:
        try:
            df = safe_read_csv(uploaded_file)
            st.sidebar.success(f"✓ Arquivo carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao ler arquivo: {e}")
            st.stop()
    else:
        st.info("👆 Faça upload de um arquivo CSV na barra lateral para começar.")
        st.stop()

    # Analyze data
    with st.spinner("📊 Analisando dados..."):
        summary = summarize_dataframe(df)
        outliers = detect_outliers(df)
        set_current_data(df, summary)

    # Quick statistics
    st.subheader("📈 Estatísticas Rápidas")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Linhas", f"{summary['n_rows']:,}")
    col2.metric("📋 Colunas", summary['n_cols'])
    col3.metric("🔢 Numéricas", len(summary.get('numeric_stats', {})))
    col4.metric("⚠️ Outliers", outliers['n_outliers'])

    # Top correlations
    if summary.get('top_correlations'):
        with st.expander("🔗 Top 5 Correlações"):
            for var1, var2, corr in summary['top_correlations'][:5]:
                col_a, col_b = st.columns([3, 1])
                col_a.write(f"{var1} ↔ {var2}")
                col_b.metric("", f"{corr:.3f}")

    # Data preview
    with st.expander("👁️ Visualizar Dados (100 primeiras linhas)"):
        st.dataframe(df.head(100), use_container_width=True)

    # Agent interaction section
    st.subheader("💬 Converse com o Agente")
    
    user_question = st.text_input(
        "Sua pergunta sobre os dados:",
        placeholder="Ex: Qual a média de Amount? (ou: Mostre a distribuição de Amount)",
        key="user_input"
    )
    
    ask_button = st.button("🚀 Perguntar ao Agente", type="primary", use_container_width=True)

    # Process question
    if ask_button and user_question.strip():
        if not st.session_state.google_api_key:
            st.error("❌ Configure sua Google API Key na barra lateral primeiro!")
            st.stop()
        
        try:
            # Create agent with selected model
            agent_executor = create_agent_executor(st.session_state.google_api_key)
            
            # Atualizar modelo no agente
            if hasattr(agent_executor.agent, 'llm_chain') and hasattr(agent_executor.agent.llm_chain, 'llm'):
                agent_executor.agent.llm_chain.llm.model = st.session_state.selected_model
            
            # Clear previous plots
            st.session_state.generated_plots = []
            
            # Run agent
            with st.spinner("🤔 O agente está pensando..."):
                result = agent_executor.invoke({"input": user_question})
            
            # Display response
            st.markdown("### 📝 Resposta do Agente:")
            st.write(result['output'])
            
            # Save to memory
            save_memory({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "question": user_question,
                "answer": result['output']
            })
            
        except Exception as e:
            st.error(f"❌ Erro ao processar pergunta: {e}")
            if "API_KEY" in str(e).upper():
                st.info("💡 Verifique se sua API Key está correta e ativa.")

    # Display generated plots
    if st.session_state.generated_plots:
        st.markdown("### 📊 Visualizações Geradas:")
        for idx, fig in enumerate(st.session_state.generated_plots):
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{idx}")

    # Conversation history
    with st.expander("📚 Histórico de Conversas (últimas 10)"):
        mem = load_memory()
        if mem:
            for entry in reversed(mem[-10:]):
                st.markdown(f"**🙋 Pergunta ({entry['timestamp'][:19]}):**")
                st.info(entry['question'])
                st.markdown(f"**🤖 Resposta:**")
                st.success(entry['answer'])
                st.markdown("---")
        else:
            st.write("Nenhuma conversa salva ainda. Faça uma pergunta para começar!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Powered by Google Gemini 🚀 | LangChain 🦜 | Streamlit ⚡</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
