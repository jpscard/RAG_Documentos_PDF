
# Aplicação RAG com Llama 3.3

Este projeto utiliza o Llama 3.8b localmente para construir uma aplicação **RAG (Recuperação Aumentada por Geração)** que permite **conversar com seus documentos**. A interface de usuário é criada com Streamlit.

 O usuário faz o upload de um documento PDF, como um artigo científico. Após o processamento do documento, ele faz perguntas ao chatbot sobre o conteúdo do artigo, como "O que é um transformador?" e "Explique o mecanismo de atenção em escala de produto de ponto". A aplicação responde corretamente a essas perguntas com base nas informações extraídas do PDF, demonstrando a capacidade do sistema de entender e consultar o documento.

## Instalação e Configuração

Siga os passos abaixo para configurar e executar o projeto.

### 1\. Configurar o Ollama

O Ollama será usado para executar o modelo Llama 3.8b localmente.

```bash
# Para configurar o Ollama no Linux
curl -fsSL https://ollama.com/install.sh | sh

# Para baixar o modelo llama3.8b
ollama pull llama3:8b 
```

### 2\. Configurar o Banco de Dados Vetorial Qdrant

O Qdrant será usado para armazenar os vetores (embeddings) dos seus documentos.

```bash
# Execute o Qdrant usando Docker
docker run -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant
```

### 3\. Instalar Dependências Python

Certifique-se de que você tem o Python 3.11 ou superior instalado.

```bash
pip install streamlit ollama llama-index-vector-stores-qdrant
```
