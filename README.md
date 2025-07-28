# Análise de Fraude em Transações Financeiras

## Introdução

Este trabalho propõe uma abordagem metodológica para análise e detecção de fraudes em transações financeiras utilizando modelos estatísticos especializados para dados de contagem com inflação de zeros. A pesquisa foca na aplicação de modelos Zero-Inflated e técnicas de seleção de variáveis stepwise para o tratamento de dados transacionais com características específicas de eventos fraudulentos.

### Problema de Pesquisa

A detecção de fraudes em sistemas financeiros apresenta desafios metodológicos específicos devido à natureza esparsa dos eventos fraudulentos em relação ao volume total de transações legítimas. Esta característica resulta em datasets com alta frequência de observações zero, violando pressupostos de distribuições tradicionais e demandando abordagens estatísticas especializadas.

### Objetivo

Desenvolver e implementar uma metodologia de análise estatística baseada em modelos Zero-Inflated para identificação de padrões em dados de fraude transacional, incluindo técnicas automatizadas de seleção de variáveis e validação de convergência.

---

## Estrutura do Trabalho

O desenvolvimento metodológico foi organizado em etapas sequenciais, cada uma documentada em notebooks específicos que compõem o pipeline analítico completo.

### 01-pre_processamento.ipynb
**Pré-processamento e Preparação dos Dados**

Este notebook documenta as etapas de aquisição, limpeza e estruturação dos dados brutos. Inclui procedimentos de:
- Carregamento e integração de datasets múltiplos
- Tratamento de inconsistências e valores ausentes
- Padronização de tipos de dados e formatos
- Criação de estruturas temporais para análise longitudinal

### 02-analise_exploratoria_dados.ipynb
**Análise Exploratória de Dados**

Apresenta a investigação descritiva das características dos dados, incluindo:
- Análise univariada e multivariada das variáveis
- Identificação de padrões temporais e espaciais
- Avaliação da distribuição da variável resposta
- Diagnóstico preliminar de adequação dos dados aos modelos propostos

### 03-modelagem_dados.ipynb
**Engenharia de Características e Transformações**

Documenta o processo de construção de variáveis preditoras através de:
- Criação de features agregadas e derivadas
- Implementação de transformações estatísticas
- Seleção preliminar de variáveis candidatas
- Preparação dos dados para modelagem estatística

### 04-modelagem_preditiva.ipynb
**Implementação de Modelos Estatísticos**

Apresenta a aplicação e comparação de modelos estatísticos especializados:
- Implementação de modelos Zero-Inflated (ZIP e ZINB)
- Aplicação de técnicas de seleção stepwise de variáveis
- Procedimentos de validação e diagnóstico de modelos
- Avaliação comparativa de abordagens metodológicas

---

## Implementação Algorítmica

### stepwise_simplified.py
Implementação otimizada do algoritmo de seleção stepwise para modelos Zero-Inflated, incluindo:
- Validação rigorosa de convergência estatística
- Interface compatível com frameworks de machine learning
- Preservação de nomenclatura original das variáveis
- Relatórios de transparência do processo de seleção

### stepwise_optimized.py
Versão original do algoritmo com funcionalidades estendidas para fins de comparação metodológica.

---

## Requisitos Técnicos

```python
numpy>=1.21.0
pandas>=1.3.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

---

## Fonte de Dados

Os dados utilizados neste estudo foram obtidos do repositório público Kaggle: [Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets), compreendendo transações financeiras sintéticas com características representativas de sistemas bancários reais.

---

## Estrutura de Execução

Para reproduzir as análises, execute os notebooks na sequência indicada:

1. `01-pre_processamento.ipynb`
2. `02-analise_exploratoria_dados.ipynb`
3. `03-modelagem_dados.ipynb`
4. `04-modelagem_preditiva.ipynb`

Cada notebook documenta completamente seus procedimentos e pode ser executado independentemente após a conclusão das etapas anteriores.

---

*Este trabalho foi desenvolvido como parte de um Trabalho de Conclusão de Curso, seguindo metodologia científica para pesquisa em análise de dados financeiros e detecção de fraudes.* 