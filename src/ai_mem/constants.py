"""Constants - Centralized configuration values for ai-mem.

This module consolidates magic numbers and configuration constants
that are used across the codebase for easier maintenance and tuning.
"""

# =============================================================================
# Token Limits
# =============================================================================

DEFAULT_TOKEN_LIMIT = 8000  # Default context token budget
MAX_TOKEN_LIMIT = 32000  # Maximum allowed token limit
MIN_TOKEN_LIMIT = 500  # Minimum useful token limit
TOKEN_BUFFER = 200  # Safety buffer for token counting
TOKENS_PER_OBSERVATION = 500  # Target tokens per compressed observation

# =============================================================================
# Search and Retrieval
# =============================================================================

DEFAULT_SEARCH_LIMIT = 50  # Default number of results to return
MAX_SEARCH_LIMIT = 1000  # Maximum results per search
MIN_SIMILARITY_SCORE = 0.3  # Minimum semantic similarity threshold
RERANK_TOP_K = 20  # Candidates to consider for reranking
FTS_BOOST_FACTOR = 1.5  # Boost factor for FTS matches

# =============================================================================
# Compression
# =============================================================================

DEFAULT_COMPRESSION_RATIO = 4.0  # Target compression ratio
MIN_COMPRESSION_RATIO = 2.0  # Minimum meaningful compression
MAX_COMPRESSION_RATIO = 10.0  # Maximum compression before quality loss
COMPRESSION_TIMEOUT_SECONDS = 30  # Timeout for compression operations

# =============================================================================
# Embedding
# =============================================================================

EMBEDDING_DIMENSION = 384  # Default embedding vector dimension
EMBEDDING_BATCH_SIZE = 100  # Batch size for async embedding
MAX_EMBEDDING_RETRIES = 3  # Retries for embedding API calls
EMBEDDING_TIMEOUT_SECONDS = 60  # Timeout for embedding operations

# =============================================================================
# Database
# =============================================================================

DB_BUSY_TIMEOUT_MS = 5000  # SQLite busy timeout
MAX_BATCH_INSERT = 500  # Maximum rows per batch insert
VACUUM_THRESHOLD_MB = 100  # Trigger vacuum when DB exceeds this size
CONTENT_HASH_LENGTH = 64  # SHA-256 hex length for deduplication

# =============================================================================
# Server
# =============================================================================

DEFAULT_SERVER_PORT = 8000  # Default API server port
MAX_REQUEST_SIZE_MB = 10  # Maximum request body size
RATE_LIMIT_REQUESTS = 100  # Default rate limit (requests per minute)
SSE_HEARTBEAT_SECONDS = 30  # Server-sent events heartbeat interval
CACHE_MAX_AGE_SECONDS = 3600  # Static content cache duration (1 hour)

# =============================================================================
# Session
# =============================================================================

SESSION_TIMEOUT_HOURS = 24  # Session expires after this many hours
MAX_SESSIONS_PER_PROJECT = 100  # Maximum sessions to track per project
SESSION_CLEANUP_DAYS = 30  # Clean up sessions older than this

# =============================================================================
# Memory Management
# =============================================================================

DECAY_HALF_LIFE_DAYS = 30  # Half-life for importance decay
CONSOLIDATION_THRESHOLD = 0.8  # Similarity threshold for consolidation
MAX_OBSERVATIONS_IN_MEMORY = 10000  # Memory cache limit
IMPORTANCE_DECAY_FACTOR = 0.95  # Per-day importance decay

# =============================================================================
# Retry and Timeout
# =============================================================================

DEFAULT_RETRY_COUNT = 3  # Default number of retries
DEFAULT_TIMEOUT_SECONDS = 30  # Default operation timeout
BACKOFF_BASE_SECONDS = 1.0  # Base for exponential backoff
BACKOFF_MAX_SECONDS = 60.0  # Maximum backoff delay

# =============================================================================
# Query Expansion
# =============================================================================

MAX_EXPANDED_TERMS = 10  # Maximum terms after expansion
SYNONYM_BOOST_FACTOR = 0.8  # Weight for synonym matches
ABBREVIATION_BOOST = 0.9  # Weight for abbreviation matches

# =============================================================================
# Knowledge Graph
# =============================================================================

ENTITY_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for entity extraction
MAX_ENTITIES_PER_OBSERVATION = 20  # Maximum entities to extract
RELATION_STRENGTH_DECAY = 0.9  # Decay factor for relation strength

# =============================================================================
# UI
# =============================================================================

OBSERVATIONS_PER_PAGE = 50  # Observations per page in UI
MAX_PREVIEW_LENGTH = 200  # Characters for observation preview
STATUS_UPDATE_INTERVAL_MS = 5000  # Status polling interval
