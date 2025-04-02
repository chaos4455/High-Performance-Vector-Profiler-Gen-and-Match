Ok, Elias Andrade, especialista sênior em documentação de projetos de software e arquiteto de sistemas da Replika IA Solutions, pronto para documentar o projeto Vectorial Profiler. 🚀

# 📚 Vectorial Profiler: Documentação Comercial e Executiva 
 
## 📝 Descrição Geral
 
O **Vectorial Profiler** é um projeto de análise e visualização de dados de perfis de jogadores, com o objetivo de facilitar o matchmaking e a identificação de comunidades com interesses similares. 🎯 A aplicação combina técnicas de *embedding*, similaridade customizada e visualização 3D interativa para apresentar uma visão abrangente do panorama de usuários, suas preferências e potenciais conexões.
 
**Componentes Principais:**
 
*   **Geração de Perfis Sintéticos:** Criação de dados de jogadores simulados para testes e demonstração.
*   **Embeddings:** Representação vetorial dos perfis para cálculo de similaridade.
*   **Índice FAISS:** Otimização da busca por vizinhos mais próximos no espaço vetorial.
*   **Score de Similaridade Customizado:** Combinação ponderada de diferentes características dos perfis para refinar a busca.
*   **Visualização 3D:** Apresentação interativa dos perfis e suas conexões utilizando *Plotly*.
*   **Frontend Web:** Interface web (Flask / FastAPI) para interação e visualização dos resultados.
*   **Monitoramento de Logs:** Dashboard em tempo real para visualização de logs da aplicação.

**Problema Resolvido:**
 
O projeto visa resolver o problema da descoberta de jogadores compatíveis em plataformas online, oferecendo uma ferramenta que vai além das buscas tradicionais por nome ou características básicas. Através da análise vetorial e da similaridade customizada, busca identificar conexões mais profundas e significativas entre os usuários. 🧑‍🤝‍🧑

## 🗂️ Estrutura do Projeto

*Esta seção detalha a organização de arquivos e diretórios, essencial para a manutenção e evolução do projeto.*

*   `.`: Diretório raiz do projeto
    *   **DOCUMENTACAO-PROJETO.md** 📝:
        *   `.\DOCUMENTACAO-PROJETO.md`
        *   Tamanho: 0.02 MB
        *   Linhas: 338
        *   Documentação do projeto (este arquivo).
    *   **data-cubic-viz-v1.py** 🧊:
        *   `.\data-cubic-viz-v1.py`
        *   Tamanho: 0.05 MB
        *   Linhas: 1152
        *   Script Python para geração da visualização 3D dos perfis.
    *   **docgenv2.py** 🐍:
        *   `.\docgenv2.py`
        *   Tamanho: 0.02 MB
        *   Linhas: 386
        *   Script Python para geração automática da documentação do projeto (README.md).
    *   **geraprofilesv1.py** ⚙️:
        *   `.\geraprofilesv1.py`
        *   Tamanho: 0.01 MB
        *   Linhas: 225
        *   Script Python para gerar perfis de jogadores sintéticos (primeira versão).
    *   **geraprofilesv2.py** 🧬:
        *   `.\geraprofilesv2.py`
        *   Tamanho: 0.05 MB
        *   Linhas: 825
        *   Script Python para gerar perfis de jogadores sintéticos (segunda versão com melhorias).
    *   **geraprofilesv3.py** 🧪:
        *   `.\geraprofilesv3.py`
        *   Tamanho: 0.08 MB
        *   Linhas: 1429
        *   Script Python para gerar perfis de jogadores sintéticos (terceira versão com otimizações e paralelização).
    *   **heathmap-data-gen-v1.py** 🔥:
        *   `.\heathmap-data-gen-v1.py`
        *   Tamanho: 0.02 MB
        *   Linhas: 450
        *   Script Python para gerar visualizações de similaridade em formato de heatmap.
    *   **heathmap-data-gen-v2.py** 🌡️:
        *   `.\heathmap-data-gen-v2.py`
        *   Tamanho: 0.02 MB
        *   Linhas: 418
        *    Script Python para gerar visualizações de similaridade em formato de heatmap (segunda versão com layout horizontal).
    *   **log-dashboard-real-time-v1.py** 📈:
        *   `.\log-dashboard-real-time-v1.py`
        *   Tamanho: 0.03 MB
        *   Linhas: 682
        *   Script Python para monitorar logs em tempo real (primeira versão).
    *   **log-dashboard-real-time-v2.py** 📊:
        *   `.\log-dashboard-real-time-v2.py`
        *   Tamanho: 0.04 MB
        *   Linhas: 850
        *   Script Python para monitorar logs em tempo real (segunda versão com melhorias).
    *   **log-dashboard-real-time-v3.py** 📉:
        *   `.\log-dashboard-real-time-v3.py`
        *   Tamanho: 0.05 MB
        *   Linhas: 983
        *   Script Python para monitorar logs em tempo real (terceira versão com mais recursos).
    *   **match-profilerv1.py** 🧑‍💻:
        *   `.\match-profilerv1.py`
        *   Tamanho: 0.01 MB
        *   Linhas: 157
        *   Script Python para realizar o match de perfis (primeira versão).
    *   **match-profilerv2-web-dash-full.py** 🌐:
        *   `.\match-profilerv2-web-dash-full.py`
        *   Tamanho: 0.03 MB
        *   Linhas: 643
        *   Script Python com a aplicação web (Flask) para o match de perfis (segunda versão).
    *   **match-profilerv3-web-dash-full-themes-fastapi.py** 🎨:
        *   `.\match-profilerv3-web-dash-full-themes-fastapi.py`
        *   Tamanho: 0.09 MB
        *   Linhas: 1724
        *   Script Python com a aplicação web (FastAPI) para o match de perfis (terceira versão com temas).
    *   **match-profilerv3-web-dash-full-themes.py** 🌈:
        *   `.\match-profilerv3-web-dash-full-themes.py`
        *   Tamanho: 0.09 MB
        *   Linhas: 1551
        *   Script Python com a aplicação web (Flask) para o match de perfis (terceira versão com temas).
    *  **requirements.txt** 📄:
        *   `.\requirements.txt`
        *   Tamanho: 0.00 MB
        *   Linhas: 17
        *   Lista de dependências Python do projeto.
    *   **test-v1-match-profilerv3-web-dash-full-themes.py** 🧪 :
        *   `.\test-v1-match-profilerv3-web-dash-full-themes.py`
        *   Tamanho: 0.07 MB
        *   Linhas: 1309
        *   Script Python com os testes automatizados do projeto.
    *   **vectorizerv1.py** 📐:
        *   `.\vectorizerv1.py`
        *   Tamanho: 0.00 MB
        *   Linhas: 69
        *   Script Python para vetorizar perfis (primeira versão).
*   `.\dashboard_logs`: Diretório para armazenar os logs do dashboard de monitoramento.
    *   **dashboard_monitor_8444.log** 📝:
        *   `.\dashboard_logs\dashboard_monitor_8444.log`
        *   Tamanho: 0.02 MB
        *   Linhas: 167
        *   Arquivo de log do dashboard de monitoramento.
*   `.\data-dash-viewer`: Diretório para salvar as visualizações 3D dos perfis.
    *   **profile\_\<ID\>\_neighbors\_score\_\<TIMESTAMP\>\_\<HASH\>.html**:
        *   Arquivos HTML contendo as visualizações 3D interativas dos perfis e seus vizinhos similares.
*   `.\databases_v3`: Diretório para armazenar os bancos de dados SQLite.
    *   **clusters\_perfis\_v3.db**: Banco de dados SQLite contendo informações sobre os clusters dos perfis.
    *   **embeddings\_perfis\_v3.db**: Banco de dados SQLite contendo os embeddings dos perfis.
    *   **perfis\_jogadores\_v3.db**: Banco de dados SQLite contendo os dados dos perfis de jogadores.
    *   **vetores\_perfis\_v3.db**: Banco de dados SQLite contendo os vetores de características dos perfis.
*   `.\img-data-outputs`: Diretório para salvar as imagens geradas pelos scripts de heatmap.
    *   **similarity\_viz\_horizontal\_origin\_\<ID\>\_\<TIMESTAMP\>\_\<HASH\>.png**: Arquivos PNG contendo as visualizações de similaridade em formato de heatmap.
*   `.\logs_v3`: Diretório para armazenar os logs dos scripts de geração de dados e visualização.
    *   Arquivos de log com timestamps indicando a data e hora de execução dos scripts.
*   `.\test-api-flask-log`: Diretório para salvar os resultados dos testes automatizados.
    *   **test\_results\_\<TIMESTAMP\>\_\<HASH\>.json**: Arquivos JSON contendo os resultados dos testes.
*   `.\valuation_v3`: Diretório para salvar os dados de valuation.
    *   **valuation\_\<TIMESTAMP\>\_origem\_\<ID\>\_scored.json**: Arquivos JSON contendo os dados de valuation dos perfis.
    *   Arquivos de log com timestamps indicando a data e hora de execução dos scripts.
* `.\valuation_v3_web_log`: Diretório para armazenar os logs da aplicação web.
    *   Arquivos de log com timestamps indicando a data e hora de execução da aplicação web.

## ⚙️ Detalhes Técnicos e Arquiteturais

O projeto Vectorial Profiler adota uma arquitetura modular, com os seguintes componentes principais:

*   **Data Layer:** Responsável pela persistência e acesso aos dados dos perfis, embeddings e clusters. Utiliza bancos de dados SQLite para armazenamento local e otimização de queries.
*   **Service Layer:** Implementa a lógica de negócio do projeto, incluindo a geração de embeddings, o cálculo de similaridade e a busca por vizinhos. Utiliza bibliotecas como `faiss` e `numpy` para otimizar o desempenho.
*   **Presentation Layer:** Fornece a interface de usuário para interação com o projeto. Implementada com Flask ou FastAPI para criar um dashboard web interativo.
*   **Visualization Layer:** Utiliza bibliotecas como Plotly e Matplotlib para gerar visualizações 3D interativas dos perfis e suas conexões.

**Tecnologias Utilizadas:**

*   **Python:** Linguagem principal de desenvolvimento.
*   **SQLite:** Banco de dados para armazenamento local dos dados.
*   **FAISS:** Biblioteca para busca eficiente de similaridade em grandes conjuntos de dados.
*   **NumPy:** Biblioteca para computação numérica.
*   **Sentence Transformers:** Biblioteca para gerar embeddings de texto.
*   **Plotly:** Biblioteca para criação de visualizações 3D interativas.
*   **Flask/FastAPI:** Frameworks para criação de aplicações web.
*   **Tailwind CSS:** Framework CSS para estilização do frontend.

## 🖥️ Como Executar e Configurar o Projeto

Siga estas instruções para configurar o ambiente e executar o projeto:

1.  **Instalar o Python:** Certifique-se de ter o Python 3.6 ou superior instalado.
2.  **Criar um ambiente virtual (opcional, mas recomendado):**
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    venv\Scripts\activate.bat  # No Windows
    ```
3.  **Instalar as dependências:**
    
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configurar as variáveis de ambiente:**
    *   Verifique se as variáveis `DATABASE_PROFILES`, `DATABASE_EMBEDDINGS` e `VALUATION_DIR` estão corretamente configuradas nos scripts Python.
    *   Ajuste os caminhos para os bancos de dados e diretórios conforme necessário.
5.  **Executar os scripts:**
    *   Para gerar os perfis sintéticos, execute o script `geraprofilesv3.py`:
    
    ```bash
    python ./geraprofilesv3.py
    ```
    *   Para executar a aplicação web, execute o script `match-profilerv3-web-dash-full-themes-fastapi.py` ou `match-profilerv3-web-dash-full-themes.py` (dependendo da versão que você deseja utilizar).

    ```bash
    python ./match-profilerv3-web-dash-full-themes-fastapi.py
    # ou
    python ./match-profilerv3-web-dash-full-themes.py
    ```

6. **Acessar o dashboard:**
    *   Abra o navegador e acesse `http://127.0.0.1:<PORTA>`, substituindo `<PORTA>` pela porta configurada (por padrão, 8881).
    

## ➕ Considerações Adicionais

*   **Arquitetura:** O projeto segue uma arquitetura modular, facilitando a manutenção e a extensão de suas funcionalidades.
*   **Padrões de Codificação:** O projeto busca seguir os padrões de codificação PEP 8 para garantir a legibilidade e a manutenibilidade do código.
*   **Licença:** O projeto é distribuído sob a licença MIT, permitindo a livre utilização, modificação e distribuição do código.
*   **Contribuições:** Contribuições são bem-vindas! Para contribuir, siga estas etapas:
    1.  Faça um fork do repositório.
    2.  Crie uma branch para sua feature ou correção de bug.
    3.  Implemente as alterações e adicione testes unitários.
    4.  Envie um pull request.
*   **Próximos Passos:**
    *   Implementar testes automatizados abrangentes para garantir a qualidade do código. ✅
    *   Documentar a API e as funções internas do projeto. ✅
    *   Otimizar o desempenho da busca por vizinhos utilizando técnicas de indexação mais avançadas.
    *   Adicionar suporte para diferentes tipos de visualização (ex: gráficos de dispersão, mapas de calor).
    *   Implementar um sistema de autenticação e autorização para garantir a segurança dos dados. 🛡️
*   **Notas:**
    *   O projeto ainda está em desenvolvimento e pode apresentar bugs ou instabilidades. 🚧
    *   Alguns recursos podem não estar totalmente implementados ou documentados.
    *   A documentação deste README.md é gerada automaticamente e pode conter imprecisões. 🤖

## ℹ️ Informações sobre o Ambiente de Geração

*   **Sistema Operacional:** Windows 10
*   **Data e Hora da Geração:** 2025-04-01 16:02:39
*   **Nome do Computador:** DESKTOP-REPLIKA
 
Documento criado por Elias Andrade - Replika IA Solutions. 