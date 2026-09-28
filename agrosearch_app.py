import math
import re
import unicodedata
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")

# ---------------------------------------------------------------------------
# Base de documentos
# ---------------------------------------------------------------------------
DOCUMENTOS = {
    1: "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    2: "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    3: "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    4: "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    5: "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}

# Stopwords já sem acento (a normalização roda antes da remoção)
STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "pra", "com", "sem", "sob", "sobre",
    "e", "ou", "mas", "que", "se", "ao", "aos", "ele", "ela", "eles", "elas", "seu",
    "sua", "seus", "suas", "este", "esta", "esse", "essa", "isso", "isto", "ser",
    "sao", "foi", "pode", "podem", "feito", "mais", "muito", "como", "entre",
    "durante", "ate", "tambem", "ja", "nao",
}

# Sufixos (sem acento) do stemmer simples; testados do mais longo para o mais curto
SUFIXOS = sorted(
    [
        "amente", "mente", "amento", "acao", "icao", "ucao", "idade", "adora", "ador",
        "osa", "oso", "ico", "ica", "ivo", "iva", "avel", "ivel", "ismo", "ista",
        "ando", "endo", "indo", "ado", "ada", "ido", "ida",
        "ar", "er", "ir", "a", "o", "e",
    ],
    key=len,
    reverse=True,
)


# ---------------------------------------------------------------------------
# Fase 1: pipeline de pré-processamento
# ---------------------------------------------------------------------------
def normalizar(texto):
    """Minúsculas e remoção de acentos."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokenizar(texto):
    """Quebra o texto em tokens alfanuméricos (descarta pontuação)."""
    return re.findall(r"[a-z0-9]+", texto)


def remover_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS_PT]


def stem(token):
    """Stemmer simples de remoção de sufixos para o português (implementado à mão)."""
    if len(token) <= 3:
        return token
    if token.endswith("oes"):          # ex.: adubacoes -> adubacao
        token = token[:-3] + "ao"
    elif token.endswith("s"):          # plural: lagartas -> lagarta
        token = token[:-1]
    for sufixo in SUFIXOS:
        if token.endswith(sufixo) and len(token) - len(sufixo) >= 3:
            return token[: -len(sufixo)]
    return token


def preprocessar(texto, usar_stopwords=True, usar_stemming=True):
    tokens = tokenizar(normalizar(texto))
    if usar_stopwords:
        tokens = remover_stopwords(tokens)
    if usar_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


# ---------------------------------------------------------------------------
# Fase 2: índice invertido
# ---------------------------------------------------------------------------
def construir_indice(tokens_por_doc):
    """indice[termo] = {doc_id: frequência do termo no documento}."""
    indice = {}
    for doc_id, tokens in tokens_por_doc.items():
        for termo, freq in Counter(tokens).items():
            indice.setdefault(termo, {})[doc_id] = freq
    return dict(sorted(indice.items()))


# ---------------------------------------------------------------------------
# Fase 3: TF-IDF e ranqueamento
#   TF(t, d)  = frequência de t em d / total de tokens de d
#   IDF(t)    = log10(N / df(t))
#   TF-IDF    = TF * IDF
#   Score(d)  = soma do TF-IDF de cada termo da consulta em d
# ---------------------------------------------------------------------------
def calcular_idf(indice, n_docs):
    return {termo: math.log10(n_docs / len(postings)) for termo, postings in indice.items()}


def calcular_vetores(tokens_por_doc, idf):
    """Vetor TF-IDF esparso de cada documento (usado no bônus de cosseno)."""
    vetores = {}
    for doc_id, tokens in tokens_por_doc.items():
        contagem = Counter(tokens)
        vetores[doc_id] = {t: (f / len(tokens)) * idf[t] for t, f in contagem.items()}
    return vetores


def ranquear(tokens_query, indice, tokens_por_doc, idf, vetores):
    termos = list(dict.fromkeys(t for t in tokens_query if t in indice))  # únicos, na ordem
    contagem_q = Counter(tokens_query)

    acumulado = {doc_id: 0.0 for doc_id in tokens_por_doc}
    detalhes = []
    for termo in termos:
        for doc_id, freq in indice[termo].items():
            tf = freq / len(tokens_por_doc[doc_id])
            peso = tf * idf[termo]
            acumulado[doc_id] += peso
            detalhes.append({
                "Termo": termo, "Doc": f"Doc {doc_id}",
                "TF": tf, "IDF": idf[termo], "TF-IDF": peso,
            })

    # Bônus: similaridade de cosseno entre o vetor da consulta e o vetor de cada documento
    vetor_q = {t: (contagem_q[t] / len(tokens_query)) * idf[t] for t in termos}
    norma_q = math.sqrt(sum(v * v for v in vetor_q.values()))
    cosseno = {}
    for doc_id, vetor_d in vetores.items():
        norma_d = math.sqrt(sum(v * v for v in vetor_d.values()))
        produto = sum(peso_q * vetor_d.get(t, 0.0) for t, peso_q in vetor_q.items())
        cosseno[doc_id] = produto / (norma_q * norma_d) if norma_q and norma_d else 0.0

    return acumulado, cosseno, detalhes


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Pré-processamento")
usar_stopwords = st.sidebar.checkbox("Remover stopwords", value=True)
usar_stemming = st.sidebar.checkbox("Aplicar stemming", value=True)
st.sidebar.caption("As mesmas opções valem para os documentos e para a consulta.")

tokens_por_doc = {d: preprocessar(t, usar_stopwords, usar_stemming) for d, t in DOCUMENTOS.items()}
indice = construir_indice(tokens_por_doc)
idf = calcular_idf(indice, len(DOCUMENTOS))
vetores = calcular_vetores(tokens_por_doc, idf)

st.title("🌱 AgroSearch")
st.caption("Motor de busca textual para manuais técnicos de agricultura sustentável.")
aba_pipeline, aba_indice, aba_busca = st.tabs(
    ["1️⃣ Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca e Ranking"]
)

# ---------- Fase 1 ----------
with aba_pipeline:
    st.subheader("Vocabulário")
    col1, col2 = st.columns(2)
    col1.metric("Termos únicos (configuração atual)", len(indice))
    bruto = {t for d in DOCUMENTOS.values() for t in preprocessar(d, False, False)}
    col2.metric("Termos únicos sem stopwords nem stemming", len(bruto))

    st.markdown("**Tamanho do vocabulário em cada combinação:**")
    comparacao = []
    for sw in (False, True):
        for stm in (False, True):
            vocab = {t for d in DOCUMENTOS.values() for t in preprocessar(d, sw, stm)}
            comparacao.append({"Stopwords": "sim" if sw else "não", "Stemming": "sim" if stm else "não",
                               "Termos únicos": len(vocab)})
    st.dataframe(pd.DataFrame(comparacao), hide_index=True)

    st.subheader("Passo a passo em um documento")
    doc_escolhido = st.selectbox("Documento", list(DOCUMENTOS), format_func=lambda d: f"Doc {d}")
    original = DOCUMENTOS[doc_escolhido]
    normalizado = normalizar(original)
    tokens = tokenizar(normalizado)
    sem_stop = remover_stopwords(tokens)
    com_stem = [stem(t) for t in sem_stop]

    st.write("**Texto original:**", original)
    st.write("**1. Normalização (minúsculas, sem acentos):**", normalizado)
    st.write("**2. Tokenização:**", tokens)
    st.write("**3. Sem stopwords:**", sem_stop)
    st.write("**4. Com stemming:**", com_stem)

    st.subheader("Frequência dos termos (configuração atual)")
    freq_total = Counter(t for tokens_doc in tokens_por_doc.values() for t in tokens_doc)
    st.dataframe(
        pd.DataFrame(freq_total.most_common(), columns=["Termo", "Frequência"]),
        hide_index=True, height=300,
    )

# ---------- Fase 2 ----------
with aba_indice:
    st.subheader("Índice invertido: termo → documentos")
    tabela_indice = pd.DataFrame([
        {"Termo": termo, "df": len(postings),
         "Documentos": ", ".join(f"Doc {d}" for d in sorted(postings)),
         "IDF": round(idf[termo], 4)}
        for termo, postings in indice.items()
    ])
    st.dataframe(tabela_indice, hide_index=True, use_container_width=True)

    with st.expander("Ver como JSON (termo → [IDs dos documentos])"):
        st.json({termo: sorted(postings) for termo, postings in indice.items()})

# ---------- Fase 3 ----------
with aba_busca:
    consulta = st.text_input("Consulta do técnico:", value="lagartas na soja")
    usar_cosseno = st.checkbox("Bônus: usar similaridade de cosseno")

    tokens_query = preprocessar(consulta, usar_stopwords, usar_stemming)
    st.write("**Termos da consulta após o pré-processamento:**", tokens_query)

    desconhecidos = [t for t in dict.fromkeys(tokens_query) if t not in indice]
    if desconhecidos:
        st.info(f"Termos que não aparecem em nenhum documento (ignorados no cálculo): {desconhecidos}")

    if not tokens_query:
        st.warning("A consulta ficou vazia após o pré-processamento.")
    else:
        acumulado, cosseno, detalhes = ranquear(tokens_query, indice, tokens_por_doc, idf, vetores)

        criterio = "TF-IDF acumulado"
        if usar_cosseno:
            criterio = st.radio("Ranquear por:", ["TF-IDF acumulado", "Cosseno"], horizontal=True)

        tabela = pd.DataFrame([
            {"Doc": f"Doc {d}", "TF-IDF acumulado": acumulado[d], "Cosseno": cosseno[d], "Texto": DOCUMENTOS[d]}
            for d in DOCUMENTOS
        ]).sort_values(criterio, ascending=False).reset_index(drop=True)
        tabela.insert(0, "Posição", range(1, len(tabela) + 1))
        if not usar_cosseno:
            tabela = tabela.drop(columns="Cosseno")

        if tabela.loc[0, criterio] <= 0:
            st.warning("Nenhum documento contém os termos da consulta (ou todos têm IDF zero).")
        else:
            vencedor = tabela.loc[0]
            st.success(f"🏆 Documento vencedor: **{vencedor['Doc']}** ({criterio} = {vencedor[criterio]:.4f})")

        def destacar_vencedor(linha):
            estilo = "background-color: #d4edda; color: #155724; font-weight: bold" if linha.name == 0 else ""
            return [estilo] * len(linha)

        formatos = {c: "{:.4f}" for c in ("TF-IDF acumulado", "Cosseno") if c in tabela.columns}
        st.dataframe(
            tabela.style.apply(destacar_vencedor, axis=1).format(formatos),
            hide_index=True, use_container_width=True,
        )

        with st.expander("Cálculo detalhado (TF, IDF e TF-IDF por termo e documento)"):
            if detalhes:
                st.dataframe(
                    pd.DataFrame(detalhes).style.format({"TF": "{:.4f}", "IDF": "{:.4f}", "TF-IDF": "{:.4f}"}),
                    hide_index=True, use_container_width=True,
                )
            else:
                st.write("Sem termos da consulta presentes no índice.")
            st.markdown(
                "- **TF(t, d)** = frequência de *t* em *d* ÷ total de tokens de *d*\n"
                "- **IDF(t)** = log10(N ÷ df(t)), com N = 5 documentos\n"
                "- **TF-IDF** = TF × IDF; o score do documento é a soma dos termos da consulta\n"
                "- **Cosseno** = (q · d) ÷ (‖q‖ · ‖d‖), com q e d em espaço TF-IDF"
            )
