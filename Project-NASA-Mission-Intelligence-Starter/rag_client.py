import chromadb
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory."""
    backends = {}
    current_dir = Path(".")

    candidate_dirs = [
        p for p in current_dir.iterdir()
        if p.is_dir() and ("chroma" in p.name.lower() or "db" in p.name.lower())
    ]

    for directory in candidate_dirs:
        try:
            client = chromadb.PersistentClient(path=str(directory))
            collections = client.list_collections()

            for collection in collections:
                collection_name = collection.name
                key = f"{directory.name}:{collection_name}"

                try:
                    count = client.get_collection(collection_name).count()
                except Exception:
                    count = 0

                backends[key] = {
                    "directory": str(directory),
                    "collection_name": collection_name,
                    "display_name": f"{collection_name} ({directory.name}, {count} chunks)",
                    "document_count": str(count),
                }

        except Exception as e:
            backends[f"{directory.name}:error"] = {
                "directory": str(directory),
                "collection_name": "",
                "display_name": f"{directory.name} - unavailable: {str(e)[:60]}",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend."""
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    where_filter = None

    if mission_filter and mission_filter.lower() != "all":
        where_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into clean context for the LLM."""
    if not documents:
        return ""

    context_parts = ["Retrieved NASA source material:"]

    seen = set()

    for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        if not document:
            continue

        normalized = document.strip()

        if normalized in seen:
            continue

        seen.add(normalized)

        mission = metadata.get("mission", "unknown")
        mission = mission.replace("_", " ").title()

        source = metadata.get(
            "source",
            metadata.get("file_path", "unknown source")
        )

        category = metadata.get(
            "document_category",
            metadata.get("category", "unknown")
        )
        category = str(category).replace("_", " ").title()

        source_header = (
            f"\n--- Source {index} ---\n"
            f"Mission: {mission}\n"
            f"Source: {source}\n"
            f"Category: {category}"
        )

        context_parts.append(source_header)

        max_length = 4000
        if len(normalized) > max_length:
            normalized = normalized[:max_length] + "\n[truncated]"

        context_parts.append(normalized)

    return "\n".join(context_parts)