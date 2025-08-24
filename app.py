import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Dashboard Profissional - Guilherme Gozzi", layout="wide")

# --------------------- Load data ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/salaries.csv")
    # Tipos
    df["cargo"] = df["cargo"].astype("category")
    df["nivel"] = df["nivel"].astype("category")
    df["regiao"] = df["regiao"].astype("category")
    df["genero"] = df["genero"].astype("category")
    df["remoto"] = df["remoto"].astype("category")
    return df

df = load_data()

# --------------------- Sidebar Menu ---------------------
st.sidebar.title("Menu")
aba = st.sidebar.radio(
    "Selecione uma aba",
    ["Home", "Formação & Experiência", "Skills", "Análise de Dados"],
    index=0
)

# --------------------- HOME ---------------------
if aba == "Home":
    st.title("Guilherme Linard F. R. Gozzi")
    st.write("Paraíso, São Paulo • (15) 99143-4003 • Guilinard2003@gmail.com • [LinkedIn](https://www.linkedin.com/in/GuilhermeLinard)")
    st.markdown("""
**Objetivo:** Estágio/Dev Júnior em Engenharia de Software, atuando com desenvolvimento web, APIs e banco de dados.  
Estudante de Engenharia de Software na FIAP, com experiência em desenvolvimento de software, levantamento de requisitos e testes.
Domínio em Java, ASP.NET MVC, API, SQL Server, e participação em projetos acadêmicos com foco em desenvolvimento web e metodologias ágeis.
Fluente em inglês (C2). Habilidades em liderança, trabalho em equipe e resolução de problemas.
""")
    st.success("Este dashboard foi criado para o CP1: apresentar meu perfil e realizar uma análise de dados aplicada a um problema de mercado (salários em TI no Brasil).")

# --------------------- FORMAÇÃO & EXPERIÊNCIA ---------------------
elif aba == "Formação & Experiência":
    st.title("Formação & Experiência")
    st.subheader("Formação Acadêmica")
    st.write("**FIAP – Centro Universitário** — Bacharelado em Engenharia de Software (Conclusão: Dez/2027)")
    st.subheader("Experiências Acadêmicas")
    st.markdown("""
- **Mahindra – FIAP (2024)**: Desenvolvimento de site institucional (HTML/CSS/JS), usabilidade e design responsivo; levantamento de requisitos; Agile; Git/GitHub.
- **Simulação Diplomática – Colégio Objetivo (2023-2024)**: Comunicação e trabalho em equipe em debates internacionais. Prêmio de melhor discurso (sustentabilidade global).
- **Mostras de Profissões – Colégio Objetivo (2023-2024)**: Coordenação de equipes, liderança e resolução de problemas sob pressão.
- **Trabalhos voluntários — Lar da Mamãe Clory (2018–Atual)**: Organização de eventos, coordenação de voluntários e receptivo.
""")

# --------------------- SKILLS ---------------------
elif aba == "Skills":
    st.title("Skills")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Técnicas")
        st.markdown("""
- **Linguagens**: Java, Python, JavaScript
- **Web**: HTML, CSS, ASP.NET MVC
- **Banco de Dados**: SQL Server
- **Ferramentas**: Git, GitHub
- **Testes**: Testes de software (funcionais)
""")
    with col2:
        st.subheader("Comportamentais")
        st.markdown("""
- Liderança e gestão de equipes
- Comunicação clara e objetiva
- Resolução de problemas sob pressão
- Adaptabilidade a novos desafios
- Trabalho em equipe multidisciplinar
""")

# --------------------- ANÁLISE DE DADOS ---------------------
else:
    st.title("Análise de Dados — Mercado de TI (Brasil)")
    st.caption("Dataset sintético e realista embutido em `data/salaries.csv`.")

    st.subheader("1) Apresentação dos dados e tipos de variáveis")
    st.write("""
**Problema de negócio:** entender fatores associados ao **salário em TI** (BRL), por exemplo: nível, região, gênero, experiência e trabalho remoto.
**Perguntas principais:**  
1. Qual é a média/mediana/mooda salarial?  
2. Há diferença salarial relevante entre gêneros?  
3. Qual a relação entre **anos de experiência** e **salário**?  
4. Existem diferenças por **região**, **nível** e **trabalho remoto**?
""")

    with st.expander("Ver amostra dos dados"):
        st.dataframe(df.head(20))

    st.markdown("**Tipos das variáveis:**")
    tipos = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index().rename(columns={"index":"coluna"})
    st.table(tipos)

    st.divider()

    st.subheader("2) Medidas centrais, dispersão, correlação e distribuições")
    colA, colB, colC = st.columns(3)
    media = df["salario_brl"].mean()
    mediana = df["salario_brl"].median()
    moda = df["salario_brl"].mode().iloc[0]
    with colA: st.metric("Média (BRL)", f"{media:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
    with colB: st.metric("Mediana (BRL)", f"{mediana:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
    with colC: st.metric("Moda (BRL)", f"{moda:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

    col1, col2 = st.columns(2)
    with col1:
        dp = df["salario_brl"].std(ddof=1)
        var = df["salario_brl"].var(ddof=1)
        st.write(f"**Desvio padrão:** {dp:,.2f} | **Variância:** {var:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
        fig = px.histogram(df, x="salario_brl", nbins=40, marginal="box", title="Distribuição de Salários (BRL)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(df, x="anos_experiencia", y="salario_brl", color="nivel",
                          trendline="ols", title="Experiência vs Salário por Nível")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Correlação (variáveis numéricas):**")
    corr = df[["anos_experiencia", "salario_brl"]].corr()
    st.table(corr)

    st.divider()

    st.subheader("3) Intervalos de Confiança e Testes de Hipótese")
    st.markdown("""
**Objetivo:** verificar se há **diferença significativa** entre as médias salariais de **Masculino** e **Feminino**.  
- **H0 (nula):** médias são **iguais**.  
- **H1 (alternativa):** médias são **diferentes**.  
Usamos **t-teste para duas amostras independentes** (variâncias não assumidas iguais).
""")

    grupo1 = df[df["genero"]=="Masculino"]["salario_brl"]
    grupo2 = df[df["genero"]=="Feminino"]["salario_brl"]
    t_stat, p_val = stats.ttest_ind(grupo1, grupo2, equal_var=False)

    def ci95(x):
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1)
        se = sd/np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        return mean - t_crit*se, mean + t_crit*se

    ci_all = ci95(df["salario_brl"])
    ci_m = ci95(grupo1)
    ci_f = ci95(grupo2)

    colx, coly, colz = st.columns(3)
    with colx: st.metric("p-valor (t-teste M vs F)", f"{p_val:.4f}")
    with coly: st.metric("IC 95% Média Geral", f"[{ci_all[0]:,.0f}; {ci_all[1]:,.0f}]".replace(",", "X").replace(".", ",").replace("X","."))
    with colz: st.metric("Diferença de médias (M - F)", f"{grupo1.mean()-grupo2.mean():,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

    st.markdown("""
**Interpretação:** se `p-valor < 0.05`, rejeitamos H0 e há evidência estatística de diferença entre as médias.  
Os **intervalos de confiança (95%)** para as médias mostram a incerteza das estimativas.
""")

    colg1, colg2 = st.columns(2)
    with colg1:
        fig3 = px.box(df, x="genero", y="salario_brl", points="all", title="Salário por Gênero")
        st.plotly_chart(fig3, use_container_width=True)
    with colg2:
        fig4 = px.box(df, x="nivel", y="salario_brl", color="nivel", title="Salário por Nível")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
**Extra (opcional para exploração):** selecione filtros abaixo.
""")
    with st.expander("Filtros"):
        c1, c2, c3 = st.columns(3)
        nivel = c1.multiselect("Nível", sorted(df["nivel"].unique().tolist()), default=sorted(df["nivel"].unique().tolist()))
        regiao = c2.multiselect("Região", sorted(df["regiao"].unique().tolist()), default=sorted(df["regiao"].unique().tolist()))
        remoto = c3.multiselect("Remoto", sorted(df["remoto"].unique().tolist()), default=sorted(df["remoto"].unique().tolist()))
        dff = df[df["nivel"].isin(nivel) & df["regiao"].isin(regiao) & df["remoto"].isin(remoto)]
        st.write(f"Registros filtrados: {len(dff)}")
        fig5 = px.histogram(dff, x="salario_brl", nbins=30, color="nivel", barmode="overlay", title="Distribuição (com filtros)")
        st.plotly_chart(fig5, use_container_width=True)