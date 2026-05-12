# Import the Qdrant client used to communicate with the Qdrant vector database
from qdrant_client import QdrantClient

# Import required models/classes for defining points and vector settings
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue

import os


class QdrantStorage:
    """
    A helper class to manage storing and searching vector embeddings
    inside a Qdrant vector database collection.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection='documents',
        dim: int | None = None
    ):
        """
        Initialize the Qdrant connection and create the collection
        if it does not already exist.

        Parameters:
        ----------
        url : str
            URL of the Qdrant server.

        collection : str
            Name of the collection where vectors will be stored.

        dim : int
            Dimension/size of the embedding vectors.
            Example: all-MiniLM-L6-v2 -> 384 dimensions.
        """

        if dim is None:
            dim = int(os.getenv("EMBED_DIM", "384"))

        if url is None:
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            
        if api_key is None:
            api_key = os.getenv("QDRANT_API_KEY", None)

        # Create connection to Qdrant server
        self.client = QdrantClient(url=url, api_key=api_key)

        # Store collection name
        self.collection = collection

        def _get_existing_dim() -> int | None:
            try:
                info = self.client.get_collection(self.collection)
            except Exception:
                return None

            cfg = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
            if cfg is None:
                cfg = getattr(getattr(getattr(info, "config", None), "params", None), "vectors_config", None)

            size = getattr(cfg, "size", None)
            if isinstance(size, int):
                return size

            # Some Qdrant versions return a dict keyed by vector name.
            if isinstance(cfg, dict):
                for v in cfg.values():
                    s = getattr(v, "size", None)
                    if isinstance(s, int):
                        return s
            return None

        if self.client.collection_exists(self.collection):
            existing_dim = _get_existing_dim()
            if existing_dim is not None and existing_dim != dim:
                # In dev it's usually better to recreate than fail every upsert.
                self.client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        else:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        """
        Insert or update vectors in the collection.

        Parameters:
        ----------
        ids : list
            Unique IDs for each vector/document.

        vectors : list
            List of embedding vectors.

        payloads : list
            Metadata associated with each vector.
            Example:
            {
                "text": "Document content",
                "source": "file.pdf"
            }
        """

        # Create PointStruct objects for each vector
        points = [
            PointStruct(
                id=ids[i],                 # Unique point ID
                vector=vectors[i],         # Embedding vector
                payload=payloads[i]        # Metadata
            )
            for i in range(len(ids))
        ]

        # Insert/update points in Qdrant collection
        self.client.upsert(
            self.collection,
            points=points
        )

    def search(self, query_vector, source_id: str = None, top_k: int = 5):
        """
        Search for the most similar vectors/documents.

        Parameters:
        ----------
        query_vector : list
            Embedding vector of the query text.
            
        source_id : str
            Optional source ID to filter the search results to a specific PDF.

        top_k : int
            Number of top matching results to return.

        Returns:
        -------
        dict
            {
                "contexts": [matched document texts],
                "sources": [unique document sources]
            }
        """
        
        query_filter = None
        if source_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_id)
                    )
                ]
            )

        # Perform similarity search in Qdrant.
        # Newer qdrant-client versions expose `query_points` instead of `search`.
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=query_filter,
                with_payload=True,
                limit=top_k,
            )
            results = getattr(response, "points", [])
        else:
            # Backwards compatibility for older clients.
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                with_payload=True,  # Return metadata/payload
                limit=top_k,        # Number of results
            )

        # Store retrieved document texts
        contexts = []

        # Store unique document sources
        sources = set()

        # Process search results
        for r in results:

            # Safely extract payload metadata
            payload = getattr(r, "payload", None) or {}

            # Extract stored text
            text = payload.get("text", "")

            # Extract source information
            source = payload.get("source", "")

            # Add text if available
            if text:
                contexts.append(text)

                # Add source to unique set
                sources.add(source)

        # Return formatted search results
        return {
            "contexts": contexts,
            "sources": list(sources)
        }

    def delete_points(self, source_id: str):
        """
        Delete all points associated with a specific source_id.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_id)
                    )
                ]
            )
        )