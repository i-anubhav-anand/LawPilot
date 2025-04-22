import asyncio
import argparse
import os
from pathlib import Path
import sys

from app.utils.corpus_loader import CorpusLoader

async def add_corpus(directory_path: str, corpus_type: str):
    """
    Add corpus files from a directory to the legal assistant.
    
    Args:
        directory_path: Path to directory containing corpus files.
        corpus_type: Type of corpus (e.g., 'law', 'ordinance').
    """
    loader = CorpusLoader()
    
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory not found: {directory_path}")
        sys.exit(1)
    
    print(f"Adding corpus files from {directory_path} as type {corpus_type}...")
    
    corpus_ids = await loader.load_corpus_directory(directory_path, corpus_type)
    
    if corpus_ids:
        print(f"Successfully added {len(corpus_ids)} corpus files:")
        for corpus_id in corpus_ids:
            print(f"  - {corpus_id}")
    else:
        print(f"No valid corpus files found in {directory_path}")

def main():
    parser = argparse.ArgumentParser(description="Add legal corpus files to the Legal AI Assistant.")
    parser.add_argument("directory", help="Directory containing corpus files (.txt)")
    parser.add_argument("--type", default="law", help="Type of corpus (e.g., 'law', 'ordinance')")
    
    args = parser.parse_args()
    
    asyncio.run(add_corpus(args.directory, args.type))

if __name__ == "__main__":
    main() 