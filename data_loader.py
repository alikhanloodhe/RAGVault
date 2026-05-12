from functools import lru_cache

from sentence_transformers import SentenceTransformer

# Import PDF reader from LlamaIndex
from llama_index.readers.file import PDFReader

# Import text splitter for chunking large documents
from llama_index.core.node_parser import SentenceSplitter

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Import os module to access environment variables
import os
from pathlib import Path


# ---------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------
load_dotenv()


# ---------------------------------------------------
# Embedding model configuration (local)
# ---------------------------------------------------
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
)

# all-MiniLM-L6-v2 -> 384 dimensions
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)


# ---------------------------------------------------
# Initialize text splitter
# ---------------------------------------------------
# chunk_size:
# Maximum size of each text chunk
#
# chunk_overlap:
# Overlapping characters between chunks
# to preserve context continuity
# ---------------------------------------------------
splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# ---------------------------------------------------
# Function: Load PDF and split into chunks
# ---------------------------------------------------
def load_and_chunk_pdf(path: str):
    """
    Load a PDF file and split its text into smaller chunks.

    Parameters:
    ----------
    path : str
        Path to the PDF file.

    Returns:
    -------
    list[str]
        List of text chunks extracted from the PDF.
    """

    raw_path = (path or "").strip()
    if not raw_path:
        raise ValueError("PDF path is empty")

    candidate = Path(os.path.expanduser(raw_path))
    if not candidate.is_absolute():
        # Resolve relative paths from the project directory (next to this file)
        candidate = (Path(__file__).resolve().parent / candidate).resolve()

    if not candidate.exists():
        project_dir = Path(__file__).resolve().parent
        pdfs = sorted(project_dir.glob("*.pdf"))

        def _norm(name: str) -> str:
            return "".join(ch for ch in name.lower() if ch.isalnum())

        target_norm = _norm(Path(raw_path).name)
        suggestions = [p.name for p in pdfs if _norm(p.name) == target_norm]

        # If there's an unambiguous normalized match, use it.
        if len(suggestions) == 1:
            candidate = (project_dir / suggestions[0]).resolve()
        else:
            message = (
                f"PDF not found: {candidate}\n"
                f"Provided: {raw_path!r}\n"
                f"Looked in project folder: {project_dir}\n"
            )
            if suggestions:
                message += "Possible matches:\n" + "\n".join(f"- {s!r}" for s in suggestions) + "\n"
            if pdfs:
                message += "PDFs available in project folder:\n" + "\n".join(
                    f"- {p.name}" for p in pdfs
                )
            else:
                message += "No .pdf files found in the project folder."

            raise FileNotFoundError(message)

    # Load PDF pages/documents
    docs = PDFReader().load_data(file=str(candidate))

    # Extract text from all pages
    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    # Store generated chunks
    chunks = []

    # Split each page text into smaller chunks
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks


# ---------------------------------------------------
# Function: Generate embeddings using Sentence Transformers
# ---------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert text chunks into vector embeddings using Sentence Transformers.

    Parameters:
    ----------
    texts : list[str]
        List of text chunks.

    Returns:
    -------
    list[list[float]]
        List of embedding vectors.
    """

    if not texts:
        return []

    model = _get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
    )

    # `encode` returns a numpy array (or list); ensure a JSON-serializable result.
    return [v.tolist() for v in vectors]