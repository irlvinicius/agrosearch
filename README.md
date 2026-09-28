# AgroSearch

Motor de busca textual para manuais técnicos de agricultura sustentável, com **índice invertido** e ranqueamento **TF-IDF** implementados do zero (sem scikit-learn), e interface em Streamlit.

Laboratório Prático 04 — Desafio Integrador, Tópicos Avançados: Recuperação de Informação / PLN (UNIPÊ).

## Funcionalidades

- **Pré-processamento**: normalização (minúsculas e remoção de acentos), tokenização, stopwords em português e stemming por remoção de sufixos, com checkboxes para ligar/desligar as etapas e ver o vocabulário mudar
- **Índice invertido**: estrutura termo → documentos construída em memória, exibida em tabela e em JSON
- **Ranqueamento TF-IDF**: TF, IDF e TF-IDF acumulado calculados à mão, com o documento vencedor destacado e o cálculo detalhado por termo
- **Bônus**: similaridade de cosseno entre o vetor da consulta e os vetores TF-IDF dos documentos

## Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — gerenciador de dependências
- Streamlit e pandas (apenas para a interface e a exibição de tabelas)
- Todo o resto usa apenas a biblioteca padrão do Python

## Como rodar

```
uv sync
uv run python -m streamlit run agrosearch_app.py
```

A aplicação abre em `http://localhost:8501`.

## Como funciona

| Etapa | Implementação |
|---|---|
| TF(t, d) | frequência de t em d ÷ total de tokens de d |
| IDF(t) | log10(N ÷ df(t)) |
| TF-IDF | TF × IDF |
| Score do documento | soma do TF-IDF dos termos da consulta |
| Cosseno (bônus) | (q · d) ÷ (‖q‖ · ‖d‖) |

## Estrutura

```
agrosearch/
├── agrosearch_app.py   # aplicação completa (todas as fases)
├── pyproject.toml      # dependências
├── uv.lock             # versões travadas
├── docs/               # relatório em PDF
└── README.md
```