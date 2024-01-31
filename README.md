# README do Repositório

## Visão Geral
Este repositório contém scripts Python destinados à análise de dados e execução de modelos de aprendizado de máquina no dataset "Dry Bean Dataset". Os scripts incluem funcionalidades para pré-processamento de dados, análise exploratória, otimização de hiperparâmetros de modelos de machine learning e geração de relatórios.

## Conteúdo do Repositório
- `lab01.py`: Script principal para execução dos modelos de machine learning. Inclui pré-processamento de dados, busca de hiperparâmetros, treinamento e avaliação de modelos como Árvore de Decisão, SVM, Random Forest e Redes Neurais.
- `lab02.py`: Script auxiliar para gerar relatórios em formato de texto dos resultados obtidos em cada rodada de testes.
- `lab03.py`: Script auxiliar para enviar e-mails com notificações sobre o progresso ou conclusão dos processos de machine learning.
- `cores.py`: Módulo auxiliar para formatação de saídas no console.
- `Dry_Bean_Dataset.csv`: Dataset utilizado para análise e treinamento dos modelos.

## Funcionalidades
- **Análise Exploratória de Dados**: O script `lab01.py` realiza uma análise exploratória inicial, tratando valores ausentes e convertendo colunas para o tipo de dados adequado.
- **Otimização de Hiperparâmetros**: Utiliza `RandomizedSearchCV` para encontrar os melhores hiperparâmetros para os modelos de Árvore de Decisão e SVM.
- **Treinamento e Avaliação de Modelos**: Diversos modelos de machine learning são treinados e avaliados, incluindo Árvore de Decisão, SVM, Random Forest, MLP e modelos combinados através do Bagging.
- **Geração de Relatórios**: Os resultados de cada rodada de testes são salvos em arquivos de texto para análises futuras.
- **Notificações por E-mail**: O script `lab03.py` é usado para enviar notificações por e-mail ao final de cada rodada de testes.

## Como Usar
1. **Configuração Inicial**: Certifique-se de que todas as dependências estão instaladas, conforme listadas no arquivo `requirements.txt`.
2. **Execução do Script `lab01.py`**: Este script é o ponto de entrada principal. Ele executa a análise de dados, treina os modelos e gera relatórios.
3. **Visualização dos Relatórios**: Após a execução, os relatórios gerados pelo `lab02.py` podem ser encontrados na pasta `relatorios`.

## Observações
- Os scripts utilizam uma abordagem de progresso simulado (usando `tqdm`) para a busca de hiperparâmetros, o que pode não refletir o progresso real em tempo.
- A funcionalidade de envio de e-mail requer configuração prévia do remetente e autenticação SMTP.

## Contribuições
Contribuições são bem-vindas. Por favor, envie um pull request ou abra uma issue para discutir as mudanças propostas.

