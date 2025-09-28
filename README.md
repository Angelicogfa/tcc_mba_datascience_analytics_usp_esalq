# Análise de Fraude em Transações Financeiras

## Introdução

Este trabalho propõe uma abordagem metodológica para análise e detecção de fraudes em transações financeiras utilizando modelos estatísticos especializados para dados de contagem com inflação de zeros. A pesquisa foca na aplicação de modelos Zero-Inflated e técnicas de seleção de variáveis stepwise para o tratamento de dados transacionais com características específicas de eventos fraudulentos.

### Problema de Pesquisa

A detecção de fraudes em sistemas financeiros apresenta desafios metodológicos específicos devido à natureza esparsa dos eventos fraudulentos em relação ao volume total de transações legítimas. Esta característica resulta em datasets com alta frequência de observações zero, violando pressupostos de distribuições tradicionais e demandando abordagens estatísticas especializadas.

### Objetivo

Desenvolver e implementar uma metodologia de análise estatística baseada em modelos Zero-Inflated para identificação de padrões em dados de fraude transacional, incluindo técnicas automatizadas de seleção de variáveis e validação de convergência.

### Execução

Para a reprodução deste trabalho se faz necessário executar em ordem os notebooks.

1. 01-pre_processamento.ipynb
2. 02-analise_exploratoria_dados.ipynb
3. 03-modelagem.ipynb

Para correta execução é recomendado executar esses notebooks em uma instancia do Google Colab, carregando os notebooks através do link disponível em cada arquivo, ou abrindo diretamente o repositorio do GitHub pelo Google Colab.

### Observações

O notebook 03-modelagem.ipynb possui vários logs armazenados e por isso o mesmo não é exibido corretamente acessando diretamente o mesmo pelo GitHub. Para visualiza-lo corretamente existem três opçãoes:

* Baixar o arquivo e abri-lo localmente, em um ambiente devidamente configurado
* Acessar o arquivo pelo Google Colab
* Abri-lo pelo GitHub Dev. Para executar tal ação basta prescionar o botão . (ponto) na pagina do repositorio.
  * URL: https://github.dev/Angelicogfa/tcc_mba_datascience_analytics_usp_esalq/tree/master 


*Este trabalho foi desenvolvido como parte de um Trabalho de Conclusão de Curso, seguindo metodologia científica para pesquisa em análise de dados financeiros e detecção de fraudes.* 
