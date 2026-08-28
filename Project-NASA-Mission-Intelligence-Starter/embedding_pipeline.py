#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the embedding pipeline."""

        if not openai_api_key:
            raise ValueError("OpenAI API key is required.")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and smaller than chunk_size."
            )

        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "NASA mission text embeddings",
                "embedding_model": embedding_model,
            },
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Split text into overlapping chunks with metadata."""

        if not text or not text.strip():
            return []

        text = text.strip()

        if len(text) <= self.chunk_size:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = 0
            chunk_metadata["chunk_start"] = 0
            chunk_metadata["chunk_end"] = len(text)

            return [(text, chunk_metadata)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]

            # Prefer a natural boundary near the end of the chunk.
            if end < len(text):
                search_start = max(0, len(chunk) - 200)

                possible_breaks = [
                    chunk.rfind(". ", search_start),
                    chunk.rfind("? ", search_start),
                    chunk.rfind("! ", search_start),
                    chunk.rfind("\n", search_start),
                ]

                best_break = max(possible_breaks)

                if best_break > 0:
                    chunk = chunk[: best_break + 1]
                    end = start + len(chunk)

            chunk = chunk.strip()

            if chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end

                chunks.append((chunk, chunk_metadata))
                chunk_index += 1

            if end >= len(text):
                break

            start = max(0, end - self.chunk_overlap)

        return chunks
    
    def check_document_exists(self, doc_id: str) -> bool:
        """Check whether a document ID already exists."""

        try:
            result = self.collection.get(ids=[doc_id])

            return bool(result.get("ids"))

        except Exception as e:
            logger.error(f"Error checking document {doc_id}: {e}")
            return False
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)
            
            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0
    
    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and 
                    metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            
            return file_doc_ids
            
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """Get an OpenAI embedding for text."""

        if not text or not text.strip():
            raise ValueError("Cannot create embedding for empty text.")

        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    def generate_document_id(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
    ) -> str:
        """Generate a stable document ID for a chunk."""

        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", file_path.stem)
        chunk_index = int(metadata.get("chunk_index", 0))

        safe_mission = str(mission).replace(" ", "_")
        safe_source = str(source).replace(" ", "_")

        return f"{safe_mission}_{safe_source}_chunk_{chunk_index:04d}"
    
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Enhanced metadata extraction
            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'
    
    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'
    
    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        
        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        
        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        
        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'
    
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                
                # Find only text files
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files
    
    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches.

        update_mode:
            skip    - skip existing documents
            update  - update existing documents
            replace - delete existing documents for the file and re-add
        """

        if not documents:
            return {
                "added": 0,
                "updated": 0,
                "skipped": 0,
            }

        if update_mode not in {"skip", "update", "replace"}:
            raise ValueError(
                "update_mode must be one of: skip, update, replace"
            )

        stats = {
            "added": 0,
            "updated": 0,
            "skipped": 0,
        }

        if update_mode == "replace":
            existing_ids = self.get_file_documents(file_path)

            if existing_ids:
                self.collection.delete(ids=existing_ids)

                logger.info(
                    f"Removed {len(existing_ids)} existing chunks "
                    f"for {file_path.name}"
                )

        for batch_start in range(0, len(documents), batch_size):

            batch = documents[
                batch_start: batch_start + batch_size
            ]

            texts_to_embed = []
            pending_items = []

            for text, metadata in batch:

                doc_id = self.generate_document_id(
                    file_path,
                    metadata,
                )

                exists = self.check_document_exists(doc_id)

                if exists:

                    if update_mode == "skip":
                        stats["skipped"] += 1
                        continue

                    if update_mode == "update":
                        success = self.update_document(
                            doc_id,
                            text,
                            metadata,
                        )

                        if success:
                            stats["updated"] += 1

                        continue

                texts_to_embed.append(text)

                pending_items.append(
                    (doc_id, text, metadata)
                )

            if not texts_to_embed:
                continue

            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts_to_embed,
                )

                ids_to_add = []
                texts_to_add = []
                metadatas_to_add = []
                embeddings_to_add = []

                for item, embedding_item in zip(
                    pending_items,
                    response.data,
                ):
                    doc_id, text, metadata = item

                    ids_to_add.append(doc_id)
                    texts_to_add.append(text)
                    metadatas_to_add.append(metadata)
                    embeddings_to_add.append(
                        embedding_item.embedding
                    )

                if ids_to_add:
                    self.collection.add(
                        ids=ids_to_add,
                        documents=texts_to_add,
                        metadatas=metadatas_to_add,
                        embeddings=embeddings_to_add,
                    )

                    stats["added"] += len(ids_to_add)

                    time.sleep(3)

            except Exception as e:
                logger.error(
                    f"Failed to embed batch starting at "
                    f"{batch_start}: {e}"
                )

        return stats


    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = "skip",
        batch_size: int = 50,
    ) -> Dict[str, int]:
        """Process all NASA text files and add them to ChromaDB."""

        stats = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }

        files = self.scan_text_files_only(base_path)

        for file_path in files:
            try:
                mission = self.extract_mission_from_path(file_path)

                if mission not in stats["missions"]:
                    stats["missions"][mission] = {
                        "files": 0,
                        "chunks": 0,
                        "added": 0,
                        "updated": 0,
                        "skipped": 0,
                    }

                logger.info(f"Processing: {file_path}")

                documents = self.process_text_file(file_path)

                stats["total_chunks"] += len(documents)

                file_stats = self.add_documents_to_collection(
                    documents=documents,
                    file_path=file_path,
                    batch_size=batch_size,
                    update_mode=update_mode,
                )

                stats["files_processed"] += 1
                stats["documents_added"] += file_stats["added"]
                stats["documents_updated"] += file_stats["updated"]
                stats["documents_skipped"] += file_stats["skipped"]

                mission_stats = stats["missions"][mission]

                mission_stats["files"] += 1
                mission_stats["chunks"] += len(documents)
                mission_stats["added"] += file_stats["added"]
                mission_stats["updated"] += file_stats["updated"]
                mission_stats["skipped"] += file_stats["skipped"]

            except Exception as e:
                stats["errors"] += 1

                logger.error(
                    f"Error processing {file_path}: {e}"
                )

        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""

        try:
            return {
                "collection_name": self.collection.name,
                "document_count": self.collection.count(),
                "metadata": self.collection.metadata,
                "persist_directory": self.chroma_persist_directory,
                "embedding_model": self.embedding_model,
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def query_collection(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Query the collection using an OpenAI query embedding."""

        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        try:
            query_embedding = self.get_embedding(query_text)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            return results

        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.collection.get()
            
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}
            
            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            
            # Analyze metadata
            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                # Count by mission
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                
                # Count by data type
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                
                # Count by document category
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                
                # Count by file type
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
        """Main function"""
        parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
        parser.add_argument('--data-path', default='.', help='Path to data directories')
        parser.add_argument('--openai-key', required=True, help='OpenAI API key')
        parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
        parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
        parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
        parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
        parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
        parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
        parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                        help='How to handle existing documents: skip, update, or replace')
        parser.add_argument('--test-query', help='Test query after processing')
        parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
        parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
        
        args = parser.parse_args()
        
        # Initialize pipeline
        logger.info("Initializing ChromaDB Embedding Pipeline...")
        pipeline = ChromaEmbeddingPipelineTextOnly(
            openai_api_key=args.openai_key,
            chroma_persist_directory=args.chroma_dir,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Handle delete source operation
        if args.delete_source:
            deleted_count = pipeline.delete_documents_by_source(args.delete_source)
            logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
            return
        
        # If stats only, show collection statistics and exit
        if args.stats_only:
            logger.info("Collection Statistics:")
            stats = pipeline.get_collection_stats()

            for key, value in stats.items():
                logger.info(f"{key}: {value}")

            return
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
            return
        
        # Process all data
        logger.info(f"Starting text data processing with update mode: {args.update_mode}")
        start_time = time.time()
        
        stats = pipeline.process_all_text_data(
            args.data_path,
            update_mode=args.update_mode,
            batch_size=args.batch_size,
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Print results
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Files processed: {stats['files_processed']}")
        logger.info(f"Total chunks created: {stats['total_chunks']}")
        logger.info(f"Documents added to collection: {stats['documents_added']}")
        logger.info(f"Documents updated in collection: {stats['documents_updated']}")
        logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        # Mission breakdown
        logger.info("\nMission breakdown:")
        for mission, mission_stats in stats['missions'].items():
            logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
            logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")
        
        # Collection info
        collection_info = pipeline.get_collection_info()
        logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
        logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
        
        # Test query if provided
        if args.test_query:
            logger.info(f"\nTesting query: '{args.test_query}'")
            results = pipeline.query_collection(args.test_query)
            if results and 'documents' in results:
                logger.info(f"Found {len(results['documents'][0])} results:")
                for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                    logger.info(f"Result {i+1}: {doc[:200]}...")
        
        logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
