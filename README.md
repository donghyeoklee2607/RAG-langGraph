# RAG with LangGraph

## Index

0. [Project Configuration](#0-project-configuration)
1. [RAG Architectures – Web Search RAG (Final)](#1-rag-architectures-003-web-search-rag-with-langgraph-final)
2. [Challenge – Q1: Data Preparation](#2-challenge--q1-data-preparation)
3. [Challenge – Q2: Retrieval Component](#3-challenge--q2-retrieval-component)
4. [Challenge – Q3: Generation Component](#4-challenge--q3-generation-component)
5. [Findings & Trade-offs](#5-findings--trade-offs)


This project implements and compares three RAG architectures using LangGraph:

- **001. Naive RAG with LangGraph**
- **002. Query Rewrite RAG with LangGraph**
- **003. Web Search RAG with LangGraph (Final Selected Architecture)**

<p align="center">
  <img src="images/RAG%20Architecture.png" width="500">
</p>

All architectures use a single PDF document as the initial knowledge source:
- Input document: `data/Deepseek-r1.pdf`


## 0. Project Configuration

PDF Loader Rank (The lower, the better)

| | PDFMiner | PDFPlumber | PyPDFium2 | PyMuPDF | PyPDF2 |
|----------|:---------:|:----------:|:---------:|:-------:|:-----:|
| Medical  | 1         | 2          | 3         | 4       | 5     |
| Law      | 3         | 1          | 1         | 3       | 5     |
| Finance  | 1         | 2          | 2         | 4       | 5     |
| Public   | 1         | 1          | 1         | 4       | 5     |
| Sum      | 5         | 5          | 7         | 15      | 20    |

Source: [AutoRAG Medium Blog](https://velog.io/@autorag/PDF-%ED%95%9C%EA%B8%80-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%B6%94%EC%B6%9C-%EC%8B%A4%ED%97%98#%EC%B4%9D%ED%8F%89)


```bash
pip install -r requirements.txt
```

## 1. RAG Architectures (003. Web Search RAG with LangGraph (Final))

The Web Search RAG architecture adds:
- **Query Rewrite**
- **Relevance Check (Groundedness)**
- **Web Search Fallback (Tavily)**

**1.1 State Definition**
```bash
class GraphState(TypedDict):
    question: Annotated[List[str], add_messages]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    relevance: Annotated[str, "Relevance"]
```

**1.2 Query Rewrite Node**
```bash
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Query Rewrite Prompt
with open("prompts/query_rewrite.txt", "r", encoding="utf-8") as f:
    template = f.read()

re_write_prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

question_rewriter = (
    re_write_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
)

# Query Rewrite Node
def query_rewrite(state: GraphState) -> GraphState:
    latest_question = state["question"][-1].content
    question_rewritten = question_rewriter.invoke({"question": latest_question})
    return {"question": question_rewritten}
```

**1.3 Relevance Check Node**
```bash
from tools.evaluator import GroundednessChecker

def relevance_check(state: GraphState) -> GraphState:
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        target="question-retrieval"
    ).create()

    response = question_answer_relevant.invoke(
        {"question": state["question"][-1].content, "context": state["context"]}
    )

    return {"relevance": response.score}
```

**1.4 Web Search Node**
```bash
def web_search(state: GraphState) -> GraphState:
    tavily_tool = TavilySearch()
    latest_question = state["question"][-1].content

    search_result = tavily_tool.search(
        query=latest_question,
        topic="general",
        max_results=5,
        format_output=True,
    )

    return {"context": search_result}
```


## 2. Challenge – Q1: Data Preparation 
**2.1 Document Loading**

From
```bash
loader = PDFPlumberLoader(source_uri)
docs.extend(loader.load())
```

To
```bash
parser = LlamaParse(
    result_type="markdown",
    num_workers=8,
    verbose=True,
    language="en",
)

documents = SimpleDirectoryReader(
    input_files=["data/Deepseek-r1.pdf"],
    file_extractor={".pdf": parser},
).load_data()

documents = [doc.to_langchain_format() for doc in documents]
```
**Advantage**
- Preserves Markdown structure.
- Accurately extracts tables, formulas, and multi-column layouts.

**2.2 Data Preprocessing**
- Page Filtering (Reference & Appendix Removal)
```
documents = documents[:16]
```
- Removal of <think> / <answer> Blocks
To avoid confusion with chain-of-thought style content embedded in the PDF,
```
def remove_think_and_answer_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL)
    return text
```
 
**2.3 Chunking**
```
RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
```
- Chunk size: 700 characters
- Overlap: 150 characters
This improves retrieval precision while preserving context continuity.



## 3. Challenge – Q2: Retrieval Component

**3.1 Retrieval Method**

| MODEL                  | PAGES PER DOLLAR | PERFORMANCE ON MTEB EVAL | MAX INPUT |
|------------------------|------------------|---------------------------|-----------|
| text-embedding-3-small | 62,500           | 62.3%                     | 8191      |
| text-embedding-3-large | 9,615            | 64.6%                     | 8191      |
| text-embedding-ada-002 | 12,500           | 61.0%                     | 8191      |

- Embedding: text-embedding-3-large
- Vector store: FAISS
- Top-k: 10

**3.2 Query Retrieves Relevant Documents**
- **Ensemble Retrieval** (Ensemble Retriever combining BM25 and FAISS.)
```python
bm25_retriever.k = 10
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 10})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.2, 0.8],
)
```
1) This ensures that only the 10 most lexically and semantically similar chunks are returned for each query.
2) If the retrieved context is judged not relevant, RAG switches to web search via Tavily and updates the context accordingly.

**- Reranking** : Replaces similarity-only ranking with cross-encoder semantic reranking.
<p align="left">
  <img src="images/reranker-benchmark.png" width="800">
</p>

```
compressor = FlashrankRerank(
    model="ms-marco-MultiBERT-L-12",
    top_n=10
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)
```


## 4. Challenge – Q3: Generation Component
**4.1 LLM Interface**
- Model: gpt-4.1-mini
- Temperature: 0 (deterministic behavior)

**4.2 Combining Query and Retrieved Context**
```
response = pdf_chain.invoke(
    {
        "question": latest_question,
        "context": context,
        "chat_history": messages_to_history(state["messages"]),
    }
)
```
- The LLM receives The latest user question that was written by Query Rewrite Node,
- and the retrieved context (from PDF or web search).

**4.3 How Context Is Used**
- For document-grounded questions (e.g., "What is DeepSeek-R1-Zero?"), the model uses only PDF chunks as evidence.
- For external questions (e.g., "what is CeADAR, Ireland?"), the model uses Tavily web search output as context.
- In both cases, answers are grounded in the provided context to reduce hallucinations.


## 5. Findings & Trade-offs
| Architecture / Design Choice                 | Strengths                                                      | Limitations                                                     |
| -------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| Naive RAG                                    | Simple and fast                                                | High hallucination risk                                         |
| Query Rewrite RAG                            | Improved retrieval accuracy via better-formed queries          | Cannot handle out-of-document queries on its own                |
| Web Search RAG                               | Handles external knowledge, highest robustness                 | Increased complexity and LLM/tooling calls                      |
| Chunk size 700 / overlap 150                 | Preserves full reasoning spans and long paragraphs             | Reduced retrieval granularity                                   |
| Ensemble Retrieval (BM25 + FAISS, 0.2 / 0.8) | Joint lexical + semantic coverage, robust across query types   | Higher memory usage and more compute per query                  |
| FlashRank Reranker (ms-marco-MultiBERT-L-12) | State-of-the-art semantic ranking accuracy                     | Additional latency for cross-encoder reranking                  |
| `text-embedding-3-large` for main index      | Higher retrieval quality on technical & long-form content      | More expensive per token than 3-small                           |
| Temperature = 0                              | High reproducibility, stable evaluation                        | Lower creativity; less suitable for open-ended generation       |
| Removal of `<think>/<answer>` CoT blocks     | Prevents contamination from embedded CoT that confuses the LLM | Some potentially useful reasoning traces are discarded entirely |
| Page filtering (exclude references/appendix) | Focuses retrieval on core paper content                        | Citations/appendix details are not retrievable                  |




