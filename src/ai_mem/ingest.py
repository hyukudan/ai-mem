import os
import pathspec
from pathlib import Path
from typing import List, Generator
from .memory import MemoryManager

def load_gitignore(root_path: Path) -> pathspec.PathSpec:
    gitignore = root_path / ".gitignore"
    patterns = []
    if gitignore.exists():
        with open(gitignore, "r") as f:
            patterns = f.read().splitlines()
    
    # Always ignore .git and internal db
    patterns.extend([".git", ".ai_mem_db", "__pycache__", "*.pyc", "node_modules", ".env", ".venv", "venv"])
    
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

def is_text_file(file_path: Path) -> bool:
    """Simple heuristic to check if a file is text."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError,  IOError):
        return False

def ingest_project(root_path: str, manager: MemoryManager, dry_run: bool = False) -> int:
    root = Path(root_path).resolve()
    spec = load_gitignore(root)
    
    files_processed = 0
    
    # Start a special ingestion session unless this is a dry run
    if not dry_run:
        manager.start_session(project=str(root), goal="Project Ingestion")
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter directories in place
        dirnames[:] = [d for d in dirnames if not spec.match_file(os.path.relpath(os.path.join(dirpath, d), root))]
        
        for filename in filenames:
            file_path = Path(dirpath) / filename
            rel_path = file_path.relative_to(root)
            
            if spec.match_file(str(rel_path)):
                continue
                
            if not is_text_file(file_path):
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                if not content.strip():
                    continue

                # Add file content to memory
                # We prefix with "FILE: <path>" to help semantic search
                if not dry_run:
                    manager.add_observation(
                        content=f"FILE: {rel_path}\n\n{content}",
                        obs_type="file_content",
                        project=str(root),
                        tags=["file", str(rel_path)],
                        summarize=False,
                    )
                files_processed += 1
                
            except Exception as e:
                print(f"Skipping {rel_path}: {e}")
                
    if not dry_run:
        manager.close_session()
    return files_processed
