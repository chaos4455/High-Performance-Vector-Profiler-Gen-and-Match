```markdown
# 🎮 Vectorial Profiler Project - Documentação Interna 🚀

## 📜 Descrição Geral

O Projeto Vectorial Profiler, criado por Elias Andrade da Replika IA Solutions, visa fornecer uma ferramenta de **visualização e análise de perfis de jogadores**, permitindo identificar padrões e similaridades entre eles. O objetivo é criar uma experiência de matchmaking mais inteligente e personalizada, facilitando a descoberta de parceiros de jogo compatíveis. O sistema utiliza **embeddings** (representações vetoriais) dos perfis para realizar buscas eficientes e um score de similaridade customizado para refinar os resultados.

**Componentes Principais:**

*   **Gerador de Perfis:** Criação de perfis sintéticos de jogadores com características diversas.
*   **Indexação FAISS:** Utilização de embeddings e da biblioteca FAISS para busca rápida de vizinhos similares.
*   **Cálculo de Similaridade Customizado:** Definição de um score ponderado para avaliar a compatibilidade entre perfis, com foco em plataformas e disponibilidade.
*   **Interface Web (Flask/FastAPI):** Exposição dos resultados em um dashboard interativo com visualizações 3D e temas customizáveis.

**Problema Resolvido:**

O projeto busca resolver a dificuldade de encontrar parceiros de jogo adequados, superando as limitações de sistemas de matchmaking tradicionais que se baseiam apenas em critérios superficiais (nível, ranking, etc.). Ao considerar características mais sutis dos jogadores (estilos de jogo, disponibilidade, plataformas), o Vectorial Profiler visa aumentar a satisfação e a retenção dos usuários.

## 🗂️ Estrutura do Projeto

*   `.`: (Diretório raiz) Contém os scripts principais do projeto.
    *   `data-cubic-viz-v1.py`: 🐍 Script Python responsável por gerar a visualização 3D dos perfis e suas similaridades, utilizando PCA para redução de dimensionalidade e Plotly para a plotagem interativa.
    *   `docgenv2.py`: 🐍 Script Python para gerar documentação do projeto.
    *   `estrutura_projeto_projeto vectorial profiler_win11pc_20250401_235515.yaml`: ⚙️ Arquivo YAML contendo a estrutura do projeto.
    *   `geraprofilesv1.py`: 🐍 Script Python para gerar perfis de jogadores e inserir no banco de dados.
    *   `geraprofilesv2.py`: 🐍 Script Python para gerar perfis de jogadores, incluindo embeddings e clusters.
    *   `geraprofilesv3.py`: 🐍 Script Python para gerar dados de perfil, vetores, embeddings e informações de cluster.
    *   `heathmap-data-gen-v1.py`: 🐍 Script Python para gerar dados para um heatmap de similaridade entre perfis.
    *   `heathmap-data-gen-v2.py`: 🐍 Script Python para gerar dados para um heatmap de similaridade entre perfis com visualizações horizontais agrupadas.
    *   `match-profilerv1.py`: 🐍 Script Python para encontrar perfis similares usando embeddings e índice FAISS, salvando os resultados em JSON.
    *   `match-profilerv2-web-dash-full.py`: 🐍 Script Python para criar um dashboard web interativo com Flask para visualização de matches.
    *   `match-profilerv2-web-dash.py`: 🐍 Script Python para criar um dashboard web interativo com Flask para visualização de matches.
    *   `match-profilerv3-web-dash-full-themes-fastapi.py`: 🐍 Script Python para criar um dashboard web interativo com FastAPI.
    *   `match-profilerv3-web-dash-full-themes.py`: 🐍 Script Python para criar um dashboard web interativo com Flask.
    *   `requirements.txt`: 📜 Lista de dependências Python necessárias para executar o projeto.
    *   `test-v1-match-profilerv3-web-dash-full-themes.py`: 🧪 Script Python para testar o script principal.
    *   `vectorizerv1.py`: 🐍 Script Python para gerar embeddings usando Sentence Transformer e criar o índice FAISS.
*   `.\data-dash-viewer`: (Diretório) Armazena arquivos HTML gerados para visualização dos dados.
    *   Arquivos HTML (ex: `profile_123_neighbors_score_timestamp.html`): 📄 Arquivos HTML contendo visualizações interativas de perfis e seus vizinhos similares.
*   `.\databases_v3`: (Diretório) Contém os bancos de dados SQLite utilizados pelo projeto.
    *   `clusters_perfis_v3.db`: 🗄️ Banco de dados SQLite para armazenar informações sobre os clusters dos perfis.
    *   `embeddings_perfis_v3.db`: 🗄️ Banco de dados SQLite para armazenar os embeddings dos perfis.
    *   `perfis_jogadores_v3.db`: 🗄️ Banco de dados SQLite para armazenar os dados dos perfis dos jogadores.
    *   `vetores_perfis_v3.db`: 🗄️ Banco de dados SQLite para armazenar os vetores de características dos perfis.
*   `.\img-data-outputs`: (Diretório) Armazena imagens geradas pelo script de heatmap.
    *   Arquivos PNG (ex: `similarity_map_origin_123_timestamp.png`): 🖼️ Imagens PNG representando heatmaps de similaridade entre perfis.
*   `.\logs_v3`: (Diretório) Contém os arquivos de log gerados pelos scripts.
    *   Arquivos de log (ex: `3d_visualization_generator_timestamp.log`): 📝 Arquivos de texto contendo informações de execução dos scripts.
*   `.\test-api-flask-log`: (Diretório) Logs dos testes da API Flask.
    *   `test_results_timestamp.json`: 📝 Arquivo JSON contendo os resultados dos testes automatizados.
*   `.\valuation_v3`: (Diretório) Contém os arquivos JSON de valuation gerados pelos scripts.
    *   Arquivos JSON (ex: `valuation_timestamp_origem_123_scored.json`): 📄 Arquivos JSON contendo os resultados da valuation, incluindo o perfil de origem e os perfis similares.
*   `.\valuation_v3_web_log`: (Diretório) Contém arquivos de log específicos do dashboard web.
    *   Arquivos de log (ex: `matchmaking_dashboard_timestamp.log`): 📝 Arquivos de texto contendo informações de execução do dashboard web.

## ⚙️ Detalhes Técnicos e Arquiteturais

### 🐍 Código Fonte (Python)

O projeto é implementado principalmente em Python, utilizando diversas bibliotecas para manipulação de dados, machine learning e construção da interface web.

*   **Bibliotecas:**
    *   `sqlite3`: Acesso aos bancos de dados SQLite.
    *   `numpy`: Manipulação de arrays e matrizes numéricas.
    *   `faiss`: Busca eficiente de vizinhos mais próximos em espaços de alta dimensão.
    *   `plotly`: Criação de gráficos interativos.
    *   `flask` ou `fastapi` : Criação da interface web.
    *   `scikit-learn`: Redução de dimensionalidade (PCA) e análise de dados (clustering). (Dependência opcional)
    *   `sentence-transformers`: (Em versões anteriores) Geração de embeddings de texto.

*   **Scripts Principais:**

    *   `data-cubic-viz-v1.py`:

        *   Gera uma visualização 3D interativa dos perfis e suas similaridades.
        *   Utiliza PCA (Principal Component Analysis) para reduzir a dimensionalidade dos embeddings e Plotly para criar a plotagem 3D.
        *   Define um tema escuro para a plotagem, com cores personalizadas para o fundo, texto, grade e marcadores.
        *  Usa a biblioteca FAISS para buscar os vizinhos mais próximos (perfis semelhantes) de um perfil de origem.
        *   Funções Principais:
            *   `parse_arguments()`: Processa os argumentos da linha de comando. Permite configurar o script, incluindo o ID do perfil de origem, o número de vizinhos a serem exibidos, a dimensão dos embeddings, os diretórios dos bancos de dados e o diretório de saída.
            *   `carregar_perfil_por_id_cached()`: Carrega os dados de um perfil a partir do banco de dados, utilizando um cache para evitar consultas repetidas.
            *   `calculate_custom_similarity()`: Calcula a similaridade personalizada entre dois perfis, com base em plataformas, disponibilidade, jogos e estilos.
            *   `load_embeddings_and_map()`: Carrega os embeddings dos perfis a partir do banco de dados e constrói o índice FAISS.
            *   `reduce_dimensionality()`: Reduz a dimensionalidade dos embeddings usando PCA.
            *   `create_3d_plot()`: Cria a figura 3D com Plotly, colorindo os vizinhos por score de similaridade e configurando os tooltips.
            *   `generate_html_file()`: Salva a figura Plotly em um arquivo HTML interativo.
            *   `main()`: Orquestra todo o processo, desde o carregamento dos dados até a geração do arquivo HTML.
        *   A arquitetura segue um fluxo de processamento linear: carregamento de dados, cálculo de similaridades, redução de dimensionalidade e visualização.

    *   `geraprofilesv3.py`:

        *   Responsável por gerar perfis de jogadores sintéticos e populados em um banco de dados SQLite. Este script implementa geração de características de perfil, vetorização, criação de embeddings e clustering, tudo com otimizações para desempenho e escalabilidade.
        *   A geração dos perfis é feita de forma procedural. Ele escolhe aleatoriamente valores para diferentes características do perfil, como idade, cidade, sexo, jogos favoritos, plataformas, etc., a partir de listas predefinidas ou distribuições de probabilidade. O processo é otimizado usando `executemany` para inserção em lote no banco de dados.
        *   A geração de vectores e embeddings simula dados reais, preenchendo com valores aleatórios ou transformações simples baseadas em características de perfil.
        *   Usa a biblioteca FAISS para indexação eficiente dos embeddings.

    *   `match-profilerv2-web-dash-full-themes-fastapi.py`:

        *   Implementa a interface web do projeto, utilizando o framework FastAPI.
        *   Responsável por carregar os dados dos perfis, realizar a busca de vizinhos similares e exibir os resultados em um dashboard interativo.
        *   Oferece suporte a temas customizáveis, permitindo alterar a aparência do dashboard.
        *   O código é bem modularizado, com funções separadas para o carregamento dos dados, o cálculo da similaridade e a construção do índice FAISS.
        *   Utiliza threading para realizar o carregamento dos dados em background, evitando o bloqueio da interface web.
        *   Funções Principais:
            *   `carregar_perfil_por_id_cached()`: Carrega os dados de um perfil a partir do banco de dados, utilizando um cache para evitar consultas repetidas.
            *    `load_data_and_build_index()`: Carrega os embeddings dos perfis a partir do banco de dados e constrói o índice FAISS.
            *    `buscar_e_rankear_vizinhos()`: Busca os vizinhos mais próximos de um perfil de origem, calcula o score de similaridade e rankeia os resultados.
            *     `index()`: Função principal da rota `/`, responsável por orquestrar todo o processo e renderizar o template HTML.

*    Arquitetura:
     *  O script `data-cubic-viz-v1.py` segue um fluxo de processamento linear: carregamento de dados, cálculo de similaridades, redução de dimensionalidade e visualização.
     *  O script `geraprofilesv3.py` segue uma arquitetura modularizada, com funções bem definidas para cada etapa do processo: geração de perfis, vetorização, criação de embeddings, clustering e salvamento dos dados.
     *  O script `match-profilerv2-web-dash-full-themes-fastapi.py` adota uma arquitetura MVC (Model-View-Controller), com os modelos representados pelos dados dos perfis e embeddings, a view pelo template HTML e o controller pelas funções que manipulam os dados e coordenam a interação entre o modelo e a view.

### 🗄️ Bancos de Dados (SQLite)

O projeto utiliza bancos de dados SQLite para armazenar os dados dos perfis, os embeddings e os resultados do clustering.

*   **Diagrama ER:**

    *   `perfis`:
        *   `id` (INTEGER, PRIMARY KEY)
        *   `nome` (TEXT)
        *   `idade` (INTEGER)
        *   `cidade` (TEXT)
        *   `estado` (TEXT)
        *   `sexo` (TEXT)
        *   `interesses_musicais` (TEXT)
        *   `jogos_favoritos` (TEXT)
        *   `plataformas_possuidas` (TEXT)
        *   `estilos_preferidos` (TEXT)
        *   `disponibilidade` (TEXT)
        *   `interacao_desejada` (TEXT)
        *   `compartilhar_contato` (BOOLEAN)
        *   `descricao` (TEXT)
    *   `embeddings`:
        *   `id` (INTEGER, PRIMARY KEY, FOREIGN KEY referencing `perfis.id`)
        *   `embedding` (BLOB)
    *`vetores`:
        *  `id` (INTEGER, PRIMARY KEY, FOREIGN KEY referencing `perfis.id`)
        *  `vetor` (BLOB)
    *   `clusters`:
        *   `id` (INTEGER, PRIMARY KEY, FOREIGN KEY referencing `perfis.id`)
        *   `cluster_id` (INTEGER)

*   **Tabelas:**

    *   `perfis`: Armazena os dados dos perfis dos jogadores, incluindo informações pessoais, preferências e descrição.
    *   `embeddings`: Armazena os embeddings dos perfis, utilizados para a busca de vizinhos similares.
    *    `vetores`: Armazena os vetores de características de cada perfil.
    *   `clusters`: Armazena os resultados do clustering, indicando a qual cluster cada perfil pertence.

*   **Esquema das Tabelas:**

    *   `perfis`:
        *   `id` (INTEGER, PRIMARY KEY): Identificador único do perfil.
        *   `nome` (TEXT): Nome do jogador.
        *   `idade` (INTEGER): Idade do jogador.
        *   `cidade` (TEXT): Cidade do jogador.
        *   `estado` (TEXT): Estado do jogador.
        *   `sexo` (TEXT): Sexo do jogador.
        *   `interesses_musicais` (TEXT): Interesses musicais do jogador.
        *   `jogos_favoritos` (TEXT): Jogos favoritos do jogador.
        *   `plataformas_possuidas` (TEXT): Plataformas que o jogador possui.
        *   `estilos_preferidos` (TEXT): Estilos de jogo preferidos do jogador.
        *   `disponibilidade` (TEXT): Disponibilidade do jogador para jogar.
        *   `interacao_desejada` (TEXT): Tipo de interação desejada pelo jogador.
        *   `compartilhar_contato` (BOOLEAN): Indica se o jogador está disposto a compartilhar o contato.
        *   `descricao` (TEXT): Descrição do perfil do jogador.
    *   `embeddings`:
        *   `id` (INTEGER, PRIMARY KEY, FOREIGN KEY referencing `perfis.id`): Identificador único do perfil (chave estrangeira para a tabela `perfis`).
        *   `embedding` (BLOB): Embedding do perfil, representado como um array de bytes.
    *   `vetores`:
        *   `id` (INTEGER, PRIMARY KEY, FOREIGN KEY referencing `perfis.id`): Identificador único do perfil (chave estrangeira para a tabela `perfis`).
        *   `vetor` (BLOB): Vetor de características do perfil, representado como um array de bytes.
    *   `clusters`:
        *   `id` (INTEGER, PRIMARY KEY, FOREIGN KEY referencing `perfis.id`): Identificador único do perfil (chave estrangeira para a tabela `perfis`).
        *   `cluster_id` (INTEGER): Identificador do cluster ao qual o perfil pertence.

*   **Consultas SQL Importantes:**

    *   Selecionar todos os perfis:

    ```sql
    SELECT * FROM perfis;
    ```

    *   Selecionar um perfil por ID:

    ```sql
    SELECT * FROM perfis WHERE id = 123;
    ```

    *   Selecionar os embeddings de todos os perfis:

    ```sql
    SELECT id, embedding FROM embeddings;
    ```

    *   Selecionar os perfis de um determinado cluster:

    ```sql
    SELECT p.* FROM perfis p
    JOIN clusters c ON p.id = c.id
    WHERE c.cluster_id = 42;
    ```

*   Exemplo de Dados: (Tabela `perfis`)

```
| id | nome             | idade | cidade        | estado | sexo      | interesses_musicais | jogos_favoritos                 | plataformas_possuidas | estilos_preferidos | disponibilidade | interacao_desejada | compartilhar_contato | descricao                                                                                                                            |
|----|------------------|-------|---------------|--------|-----------|---------------------|---------------------------------|-----------------------|--------------------|-----------------|--------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Maria Silva      | 28    | São Paulo     | SP     | Feminino  | Pop, Eletrônica     | League of Legends, The Witcher 3 | PC, PlayStation 5    | RPG, Aventura      | Noite           | Online             | 1                   | Jogadora apaixonada por RPGs e música eletrônica, busco parceiros para jogar à noite.                                           |
| 2  | João Oliveira    | 35    | Rio de Janeiro | RJ     | Masculino | Rock, Metal         | Counter-Strike, Dota 2          | PC                    | FPS, Estratégia     | Fim de Semana   | Presencial         | 0                   | Curto jogar CS com a galera nos finais de semana, rock e cerveja!                                                               |
| 3  | Ana Souza        | 22    | Belo Horizonte | MG     | Feminino  | MPB, Indie          | Stardew Valley, Animal Crossing | Nintendo Switch       | Simulação, Aventura | Tarde           | Indiferente        | 1                   | Busco amigos para jogar Stardew Valley e relaxar.                                                                             |
| 4  | Pedro Almeida    | 41    | Porto Alegre  | RS     | Masculino | Clássica, Jazz      | Civilization VI, Cities Skylines  | PC                    | Estratégia         | Manhã           | Online             | 0                   | Estrategista que adora jogos de construção e música clássica.                                                                |
| 5  | Fernanda Costa   | 29    | Salvador      | BA     | Feminino  | Funk, Hip Hop       | Fortnite, Free Fire             | Mobile                | Battle Royale      | Madrugada       | Online             | 1                   | Viciada em Battle Royale no mobile, procuro squad para jogar de madrugada.                                                      |
```

*   **Observações sobre Otimizações de Queries:**
    *   Utilizar índices nas colunas utilizadas em cláusulas `WHERE` (ex: `id`, `cluster_id`) para acelerar as consultas.
    *   Evitar consultas `SELECT *` em tabelas muito grandes, especificando apenas as colunas necessárias.
    *   Utilizar `PRAGMA` para otimizar o desempenho do SQLite (ex: `PRAGMA journal_mode=WAL`).

### ⚙️ Configurações (JSON/YAML)

O projeto utiliza arquivos YAML para armazenar configurações importantes, como os pesos dos scores de similaridade. Os arquivos YAML são utilizados para facilitar a leitura e a edição das configurações.

*   **Exemplo de Configuração (YAML):**

    ```yaml
    WEIGHTS:
      jogos: 0.40
      estilos: 0.30
      plataformas: 0.20
      disponibilidade: 0.10
      interacao: 0.05
    ```

*   **Principais Chaves de Configuração:**
    *   `WEIGHTS`: Dicionário contendo os pesos para o cálculo do score de similaridade.
        *   `jogos`: Peso para a similaridade nos jogos favoritos.
        *   `estilos`: Peso para a similaridade nos estilos preferidos.
        *   `plataformas`: Peso para a similaridade nas plataformas possuídas.
        *   `disponibilidade`: Peso para a similaridade na disponibilidade.
        *   `interacao`: Peso para a similaridade no tipo de interação desejada.
    *   `NUM_NEIGHBORS_TARGET`: Número de vizinhos mais próximos a serem retornados.
    *    `INITIAL_SEARCH_FACTOR`: Fator de multiplicação para aumentar o número de candidatos iniciais na busca FAISS.
    *    `MIN_CUSTOM_SCORE_THRESHOLD`: Score mínimo para considerar um perfil como um match válido.

*   **Caminhos dos Arquivos de Configuração:** Os caminhos dos arquivos de configuração são definidos como variáveis globais nos scripts Python, facilitando a localização e a modificação.

*  Tamanho do arquivo: cerca de 0.77 MB
*  Número de linhas: por volta de 10308

## 🚀 Como Executar e Configurar o Projeto

1.  **Configurar o ambiente:**

    *   Certifique-se de ter o Python 3.6 ou superior instalado.
    *   Crie um ambiente virtual para isolar as dependências do projeto:

    ```bash
    python -m venv .venv
    ```

    *   Ative o ambiente virtual:

        *   No Windows:

        ```bash
        .venv\Scripts\activate
        ```

        *   No Linux/macOS:

        ```bash
        source .venv/bin/activate
        ```

2.  **Instalar as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurar os bancos de dados:**

    *   Os bancos de dados SQLite devem estar localizados no diretório `databases_v3`.
    *   Certifique-se de que os caminhos para os bancos de dados estejam corretos nos scripts Python.
    *   Execute os scripts de geração de dados (`geraprofilesv3.py`) para popular os bancos de dados.

4.  **Configurar os arquivos de configuração:**

    *   Verifique os valores das chaves de configuração nos arquivos YAML (ex: pesos dos scores de similaridade).
    *   Modifique os valores conforme necessário para ajustar o comportamento do projeto.

5.  **Executar os scripts:**

    *   Para gerar a visualização 3D:

    ```bash
    python data-cubic-viz-v1.py -id <id_do_perfil>
    ```

    *   Para executar o dashboard web (Flask):

    ```bash
    python match-profilerv2-web-dash-full-themes.py
    ```
    *   Para executar o dashboard web (FastAPI):
    ```
    uvicorn match-profilerv3-web-dash-full-themes-fastapi:app --reload
    ```
    *   Para executar os testes automatizados:

    ```bash
    python test-api-flask-log
    ```

## ➕ Considerações Adicionais

*   **Arquitetura do Projeto:** O projeto adota uma arquitetura híbrida, com elementos de MVC (Model-View-Controller) na interface web e um fluxo de processamento linear nos scripts de geração de dados e visualização.
*   **Padrões de Codificação:** O código segue as convenções de estilo do Python (PEP 8), com nomes de variáveis e funções descritivos e comentários explicativos.
*   **Licença:** A licença sob a qual o projeto é distribuído não foi explicitamente fornecida.
*   **Contribuições:** O projeto está em um estágio de desenvolvimento interno, e as contribuições externas podem não ser aceitas no momento. No entanto, sugestões e feedback são bem-vindos.
*   **Próximos Passos:**
    *   Implementar testes automatizados para garantir a qualidade do código e a estabilidade do sistema.
    *   Explorar diferentes algoritmos de machine learning para a geração de embeddings e o cálculo da similaridade.
    *   Adicionar mais opções de configuração para personalizar o comportamento do sistema.
*    **Estado do projeto:** O projeto possui uma base funcional, mas com áreas a serem trabalhadas. A interface web é funcional, mas pode ser melhorada com testes e uma validação de dados e modelos.

## 💽 Informações sobre o ambiente que o gerou

*   **Sistema Operacional:** Windows
*   **Data e Hora da geração:** 2025-04-01 23:58:37
*   **Nome do computador:** win11pc
```
