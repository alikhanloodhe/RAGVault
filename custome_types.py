# Import Pydantic for creating structured data models
import pydantic


# ---------------------------------------------------
# Model: RAGChunkAndSrc
# ---------------------------------------------------
# Represents a single document source and its chunks
# used during ingestion into the vector database.
# ---------------------------------------------------
class RAGChunkAndSrc(pydantic.BaseModel):

    # List of text chunks extracted from a document
    chunks: list[str]

    # Optional source identifier
    # Example:
    # "document1.pdf"
    # "notes.txt"
    source_id: str = None


# ---------------------------------------------------
# Model: RAGUpsertResult
# ---------------------------------------------------
# Returned after inserting/updating embeddings
# into the vector database.
# ---------------------------------------------------
class RAGUpsertResult(pydantic.BaseModel):

    # Number of chunks/documents successfully ingested
    ingested: int


# ---------------------------------------------------
# Model: RAGSearchResult
# ---------------------------------------------------
# Represents search results retrieved from
# the vector database.
# ---------------------------------------------------
class RAGSearchResult(pydantic.BaseModel):

    # Retrieved relevant text chunks/contexts
    contexts: list[str]

    # Sources associated with retrieved contexts
    # Example:
    # ["doc1.pdf", "report.pdf"]
    sources: list[str]


# ---------------------------------------------------
# Model: RAQQueryResult
# ---------------------------------------------------
# Represents the final response returned
# by the RAG pipeline after querying.
# ---------------------------------------------------
class RAQQueryResult(pydantic.BaseModel):

    # Final generated answer from the LLM
    answer: str

    # Sources used to generate the answer
    sources: list[str]

    # Number of retrieved contexts used
    num_contexts: int