"""
Dashboard — Modelo Preditivo de Ocorrências Criminais SP
Executar: streamlit run dashboard.py
"""

import os, subprocess, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── garante que os dados existem ──────────────────────────────────────────────
if not os.path.exists("data/crimes_ssp_sp.csv"):
    subprocess.run([sys.executable, "data/gerar_dados_ssp.py"], check=True)

# ── coordenadas aproximadas das delegacias ────────────────────────────────────
COORDS = {
    "Centro":          (-23.5505, -46.6333),
    "Sé":              (-23.5475, -46.6361),
    "Brás":            (-23.5436, -46.6113),
    "Lapa":            (-23.5251, -46.7021),
    "Pinheiros":       (-23.5613, -46.6972),
    "Vila Mariana":    (-23.5889, -46.6386),
    "Ipiranga":        (-23.5901, -46.6058),
    "Santo André":     (-23.6639, -46.5383),
    "Guarulhos":       (-23.4543, -46.5333),
    "Osasco":          (-23.5325, -46.7919),
    "São Bernardo":    (-23.6940, -46.5650),
    "Mauá":            (-23.6678, -46.4613),
    "Diadema":         (-23.6861, -46.6228),
    "Carapicuíba":     (-23.5218, -46.8350),
    "Taboão da Serra": (-23.6016, -46.7556),
    "Itaquera":        (-23.5380, -46.4561),
    "Penha":           (-23.5226, -46.5388),
    "Mooca":           (-23.5490, -46.5977),
    "Santana":         (-23.4924, -46.6274),
    "Campo Limpo":     (-23.6313, -46.7340),
}

# ── configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Criminalidade SP — Modelo Preditivo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── dados ─────────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    hist = pd.read_csv("data/crimes_ssp_sp.csv")
    prev = pd.read_csv("outputs/previsoes_2024.csv")
    aval = pd.read_csv("outputs/avaliacao_2024_detalhada.csv")

    hist["periodo"] = hist["ano"].astype(str) + "-" + hist["mes"].astype(str).str.zfill(2)

    for df in [hist, prev, aval]:
        df["lat"] = df["delegacia"].map(lambda d: COORDS.get(d, (0, 0))[0])
        df["lon"] = df["delegacia"].map(lambda d: COORDS.get(d, (0, 0))[1])

    return hist, prev, aval

hist, prev, aval = carregar_dados()

MESES_NOME = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
              7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
DELEGACIAS  = sorted(hist["delegacia"].unique())
ANOS        = sorted(hist["ano"].unique())
TIPOS_CRIME = ["furtos","roubos","lesao_corporal","estelionato","trafico_drogas","homicidios_dolosos"]

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Bras%C3%A3o_do_Estado_de_S%C3%A3o_Paulo.svg/120px-Bras%C3%A3o_do_Estado_de_S%C3%A3o_Paulo.svg.png", width=60)
    st.title("Criminalidade SP")
    st.caption("Modelo Preditivo · SSP-SP")
    st.divider()

    pagina = st.radio("Navegar", [
        "📊 Visão Geral",
        "🗺️ Mapa de Ocorrências",
        "📈 Análise Histórica",
        "🤖 Desempenho do Modelo",
        "🔮 Previsões 2024",
    ])

    st.divider()
    del_sel  = st.multiselect("Delegacias", DELEGACIAS, default=DELEGACIAS[:5])
    anos_sel = st.multiselect("Anos (histórico)", ANOS, default=list(ANOS))

    if not del_sel:
        del_sel = DELEGACIAS
    if not anos_sel:
        anos_sel = list(ANOS)

hist_f = hist[hist["delegacia"].isin(del_sel) & hist["ano"].isin(anos_sel)]
prev_f = prev[prev["delegacia"].isin(del_sel)]
aval_f = aval[aval["delegacia"].isin(del_sel)]

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "📊 Visão Geral":
    st.title("Visão Geral — Criminalidade SP")

    tot   = hist_f["total_ocorrencias"].sum()
    media = hist_f.groupby(["delegacia","ano","mes"])["total_ocorrencias"].sum().mean()
    pior  = hist_f.groupby("delegacia")["total_ocorrencias"].sum().idxmax()
    mape  = (aval_f["erro_%"].abs()).mean() if not aval_f.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Ocorrências", f"{tot:,.0f}")
    c2.metric("Média mensal/delegacia", f"{media:,.0f}")
    c3.metric("Delegacia mais crítica", pior)
    c4.metric("MAPE do modelo (2024)", f"{mape:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ocorrências por delegacia (total histórico)")
        tot_del = hist_f.groupby("delegacia")["total_ocorrencias"].sum().sort_values(ascending=True)
        fig = px.bar(tot_del, orientation="h",
                     labels={"value":"Total","index":"Delegacia"},
                     color=tot_del.values,
                     color_continuous_scale="Reds")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Composição por tipo de crime")
        tipos_tot = hist_f[TIPOS_CRIME].sum().sort_values(ascending=False)
        fig = px.pie(values=tipos_tot.values, names=tipos_tot.index,
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evolução anual do total de ocorrências")
    anual = hist_f.groupby(["ano","delegacia"])["total_ocorrencias"].sum().reset_index()
    fig = px.line(anual, x="ano", y="total_ocorrencias", color="delegacia",
                  markers=True, labels={"total_ocorrencias":"Ocorrências","ano":"Ano"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — MAPA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🗺️ Mapa de Ocorrências":
    st.title("Mapa de Ocorrências — Grande SP")

    col1, col2 = st.columns([2, 1])
    with col1:
        ano_mapa = st.select_slider("Ano", options=ANOS, value=ANOS[-1])
    with col2:
        tipo_mapa = st.selectbox("Tipo de crime", ["total_ocorrencias"] + TIPOS_CRIME,
                                  format_func=lambda x: x.replace("_"," ").title())

    hist_mapa = hist[hist["ano"] == ano_mapa].groupby("delegacia")[tipo_mapa].sum().reset_index()
    hist_mapa["lat"] = hist_mapa["delegacia"].map(lambda d: COORDS.get(d, (0,0))[0])
    hist_mapa["lon"] = hist_mapa["delegacia"].map(lambda d: COORDS.get(d, (0,0))[1])
    hist_mapa = hist_mapa[hist_mapa["lat"] != 0]

    fig = px.scatter_mapbox(
        hist_mapa, lat="lat", lon="lon",
        size=tipo_mapa, color=tipo_mapa,
        hover_name="delegacia",
        hover_data={tipo_mapa: True, "lat": False, "lon": False},
        color_continuous_scale="YlOrRd",
        size_max=55, zoom=9.5,
        mapbox_style="carto-positron",
        title=f"{tipo_mapa.replace('_',' ').title()} por Delegacia — {ano_mapa}",
        labels={tipo_mapa: "Ocorrências"},
    )
    fig.update_layout(height=580, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Ranking — {ano_mapa}")
    st.dataframe(
        hist_mapa[["delegacia", tipo_mapa]].sort_values(tipo_mapa, ascending=False)
        .rename(columns={"delegacia":"Delegacia", tipo_mapa:"Ocorrências"})
        .reset_index(drop=True),
        use_container_width=True, height=280,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — ANÁLISE HISTÓRICA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📈 Análise Histórica":
    st.title("Análise Histórica")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sazonalidade mensal")
        saz = hist_f.groupby("mes")["total_ocorrencias"].mean().reset_index()
        saz["mes_nome"] = saz["mes"].map(MESES_NOME)
        fig = px.bar(saz, x="mes_nome", y="total_ocorrencias",
                     labels={"mes_nome":"Mês","total_ocorrencias":"Média de Ocorrências"},
                     color="total_ocorrencias", color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Heatmap delegacia × mês")
        heat = hist_f.groupby(["delegacia","mes"])["total_ocorrencias"].mean().reset_index()
        heat_pivot = heat.pivot(index="delegacia", columns="mes", values="total_ocorrencias")
        heat_pivot.columns = [MESES_NOME[c] for c in heat_pivot.columns]
        fig = px.imshow(heat_pivot, color_continuous_scale="YlOrRd",
                        labels={"color":"Média ocorr."},
                        aspect="auto")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Série temporal por tipo de crime")
    tipo_hist = st.selectbox("Tipo", TIPOS_CRIME,
                              format_func=lambda x: x.replace("_"," ").title())
    serie = hist_f.groupby(["ano","mes"])[tipo_hist].sum().reset_index()
    serie["data"] = pd.to_datetime(serie[["ano","mes"]].assign(day=1))
    fig = px.area(serie, x="data", y=tipo_hist,
                  labels={"data":"Data", tipo_hist: tipo_hist.replace("_"," ").title()})
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlação entre variáveis")
    num_cols = ["populacao","densidade_demografica","idh"] + TIPOS_CRIME + ["total_ocorrencias"]
    corr = hist_f[num_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    text_auto=".2f", aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — DESEMPENHO DO MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Desempenho do Modelo":
    st.title("Desempenho do Modelo Preditivo")

    st.info("Ground truth calibrado com dados oficiais SSP-SP/Agência SP 2024: roubos −15%, furtos −12%, residências −27%.")

    if aval_f.empty:
        st.warning("Nenhuma delegacia selecionada.")
        st.stop()

    mae  = aval_f["erro_absoluto"].mean()
    mape = aval_f["erro_%"].abs().mean()
    vies = aval_f["erro_%"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE médio", f"{mae:.1f} ocorr./mês")
    c2.metric("MAPE médio", f"{mape:.1f}%")
    c3.metric("Viés médio", f"{vies:+.1f}%", delta_color="inverse")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Erro % por delegacia")
        err_del = aval_f.groupby("delegacia")["erro_%"].mean().sort_values()
        fig = px.bar(err_del, orientation="h",
                     color=err_del.values,
                     color_continuous_scale=["#4CAF50","#FFC107","#F44336"],
                     labels={"value":"Erro %","index":"Delegacia"})
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("MAPE por delegacia")
        mape_del = aval_f.groupby("delegacia")["erro_%"].apply(lambda x: x.abs().mean()).sort_values()
        cores = ["#4CAF50" if v < 10 else "#FFC107" if v < 25 else "#F44336" for v in mape_del]
        fig = go.Figure(go.Bar(
            x=mape_del.values, y=mape_del.index,
            orientation="h", marker_color=cores
        ))
        fig.add_vline(x=10, line_dash="dot", line_color="#4CAF50", annotation_text="10%")
        fig.add_vline(x=25, line_dash="dot", line_color="#FFC107", annotation_text="25%")
        fig.update_layout(xaxis_title="MAPE (%)", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evolução mensal — Real vs Previsto (total delegacias selecionadas)")
    mensal_real = aval_f.groupby("mes")["total_ocorrencias_real"].sum()
    mensal_prev = aval_f.groupby("mes")["previsao_ocorrencias"].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(MESES_NOME.values()), y=mensal_real.values,
                             mode="lines+markers", name="Real (ground truth)",
                             line=dict(color="#F44336", width=3)))
    fig.add_trace(go.Scatter(x=list(MESES_NOME.values()), y=mensal_prev.values,
                             mode="lines+markers", name="Previsto pelo modelo",
                             line=dict(color="#2196F3", width=3, dash="dash")))
    fig.update_layout(xaxis_title="Mês", yaxis_title="Ocorrências", height=340, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver dados detalhados de avaliação"):
        st.dataframe(aval_f.sort_values("erro_%", ascending=False), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — PREVISÕES 2024
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔮 Previsões 2024":
    st.title("Previsões 2024")

    tot_prev = prev_f["previsao_ocorrencias"].sum()
    del_maior = prev_f.groupby("delegacia")["previsao_ocorrencias"].sum().idxmax()
    mes_pico  = prev_f.groupby("mes")["previsao_ocorrencias"].sum().idxmax()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total previsto 2024", f"{tot_prev:,.0f}")
    c2.metric("Delegacia mais crítica", del_maior)
    c3.metric("Mês com mais ocorrências", MESES_NOME[mes_pico])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total previsto por delegacia")
        tot_del = prev_f.groupby("delegacia")["previsao_ocorrencias"].sum().sort_values(ascending=True)
        fig = px.bar(tot_del, orientation="h",
                     color=tot_del.values, color_continuous_scale="Oranges",
                     labels={"value":"Ocorrências previstas","index":"Delegacia"})
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição mensal prevista")
        mensal = prev_f.groupby("mes")["previsao_ocorrencias"].sum().reset_index()
        mensal["mes_nome"] = mensal["mes"].map(MESES_NOME)
        fig = px.bar(mensal, x="mes_nome", y="previsao_ocorrencias",
                     color="previsao_ocorrencias", color_continuous_scale="Oranges",
                     labels={"mes_nome":"Mês","previsao_ocorrencias":"Ocorrências"})
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Série mensal por delegacia")
    fig = px.line(prev_f.sort_values("mes"), x="mes", y="previsao_ocorrencias",
                  color="delegacia", markers=True,
                  labels={"mes":"Mês","previsao_ocorrencias":"Ocorrências Previstas"})
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=list(MESES_NOME.values()))
    fig.update_layout(height=380, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela completa de previsões"):
        tabela = prev_f.groupby(["delegacia","mes"])["previsao_ocorrencias"].sum().reset_index()
        tabela["mes"] = tabela["mes"].map(MESES_NOME)
        tabela.columns = ["Delegacia","Mês","Previsão"]
        st.dataframe(tabela, use_container_width=True)

# ── rodapé ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Fonte dos dados históricos: dataset sintético baseado na estrutura SSP-SP | Ground truth 2024 calibrado com dados oficiais Agência SP / SSP-SP")
