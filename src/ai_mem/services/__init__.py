"""Services - Modular components extracted from MemoryManager.

This package contains focused service classes that handle specific
responsibilities, following the Single Responsibility Principle.

Services:
- ListenerService: Manage observation listeners/callbacks
- SearchCacheService: Manage search result caching
- SimilarityService: Compute text and embedding similarities
- RerankingService: Rerank search results
- SessionService: Manage session lifecycle
- ObservationService: Create, store, retrieve observations
- SearchService: Orchestrate search operations
- TagService: Manage observation tags
- ProjectService: Manage projects and stats
- ConsolidationService: Find and consolidate similar observations
- UserMemoryService: User-scoped memory operations
- IndexingService: Handle vector indexing
- AuthService: User authentication and JWT tokens
"""

from .listener import ListenerService
from .cache import SearchCacheService
from .similarity import SimilarityService
from .reranking import RerankingService
from .session import SessionService
from .observation import ObservationService
from .search import SearchService
from .tag import TagService
from .project import ProjectService
from .consolidation import ConsolidationService
from .user_memory import UserMemoryService
from .indexing import IndexingService
from .auth import AuthService

__all__ = [
    "ListenerService",
    "SearchCacheService",
    "SimilarityService",
    "RerankingService",
    "SessionService",
    "ObservationService",
    "SearchService",
    "TagService",
    "ProjectService",
    "ConsolidationService",
    "UserMemoryService",
    "IndexingService",
    "AuthService",
]
