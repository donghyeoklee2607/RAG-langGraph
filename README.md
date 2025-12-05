# RAG with LangGraph

This project implements and compares three RAG architectures using LangGraph:

- **001. Naive RAG with LangGraph**
- **002. Query Rewrite RAG with LangGraph**
- **003. Web Search RAG with LangGraph (Final Selected Architecture)**
![RAG Architecture](images/RAG%20Architecture.png)


All architectures use a single PDF document as the initial knowledge source:
- Input document: `data/Deepseek-r1.pdf`


## 0. Project Configuration

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
        max_results=6,
        format_output=True,
    )

    return {"context": search_result}
```


## 2. Challenge – Q1: Data Preparation 
**2.1 Document Loading**

```bash
loader = PDFPlumberLoader(source_uri)
docs.extend(loader.load())
```

Each page is converted into a Document object.

**2.2 Chunking**
```
RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
```
- Chunk size: 300 characters
- Overlap: 50 characters
This improves retrieval precision while preserving context continuity.

**2.3 Text Normalisation**
- Although techniques like stopword removal, lower-casing, and punctuation cleaning are possible, they were intentionally not applied.
- Because the source is a technical document with formulas, abbreviations, and domain-specific terms, strong normalisation could distort meaning.
- Therefore, the raw text extracted from the PDF is preserved.


## 3. Challenge – Q2: Retrieval Component

**3.1 Retrieval Method**
All three architectures use dense vector search:
- Embedding: text-embedding-3-small
- Vector store: FAISS
- Top-k: 10

**3.2 Query Retrieves Relevant Documents**
- The system retrieves DeepSeek-related chunks from data/Deepseek-r1.pdf.
- These chunks are concatenated and used as context for the LLM.

```python
def create_retriever(self, vectorstore):
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": self.k}   # Top-k = 10
    )
    return dense_retriever
```
- This ensures that only the 10 most semantically similar chunks are returned for each query.

- If the retrieved context is judged not relevant, RAG switches to web search via Tavily and updates the context accordingly.


## 4. Challenge – Q3: Generation Component
**4.1 LLM Interface**
- Model: gpt-4o-mini
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

- The LLM receives The latest user question,
- the retrieved context (from PDF or web search),
- and Chat history

**4.3 How Context Is Used**
- For document-grounded questions (e.g., "What is DeepSeek-R1-Zero?"), the model uses only PDF chunks as evidence.
- For external questions (e.g., "what is CeADAR, Ireland?"), the model uses Tavily web search output as context.
- In both cases, answers are grounded in the provided context to reduce hallucinations.


## 5. Findings & Trade-offs
| Architecture / Design Choice | Strengths                                      | Limitations                                              |
|------------------------------|-----------------------------------------------|----------------------------------------------------------|
| Naive RAG                    | Simple and fast                               | High hallucination risk                                  |
| Query Rewrite RAG            | Improved retrieval accuracy                   | Cannot handle out-of-document queries                    |
| Web Search RAG               | Handles external knowledge, highest robustness| Increased complexity and LLM calls                       |
| Small chunks                 | High retrieval precision                      | Larger vector index                                      |
| Temperature = 0              | High reproducibility                          | Lower creativity                                         |
| No text normalisation        | Preserves technical terms and original meaning| Potential noise from stopwords and unnormalised tokens   |


