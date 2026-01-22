"""Integration with ai-skills for expert knowledge enrichment.

This module integrates ai-skills (https://github.com/user/ai-skills) to combine:
- Persistent memory (ai-mem observations)
- Expert knowledge (ai-skills 80+ best practices)

The integration provides:
- Unified search across memory and skills
- Automatic skill suggestions based on queries
- Enriched observations with relevant skills
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ai_mem.integrations.skills")

# Try to import ai-skills (optional dependency)
try:
    from aiskills.core.router import get_router as get_skill_router
    from aiskills.core.registry import SkillRegistry
    from aiskills.core.manager import SkillManager

    AISKILLS_AVAILABLE = True
    logger.info("ai-skills integration available")
except ImportError:
    AISKILLS_AVAILABLE = False
    logger.debug("ai-skills not installed - integration disabled")


@dataclass
class SkillSuggestion:
    """A suggested skill based on a query or observation."""

    name: str
    description: str
    category: str
    relevance_score: float
    tokens_estimate: int
    tags: List[str]


@dataclass
class EnrichedSearchResult:
    """Combined search result from memory + skills."""

    # Memory observations
    observations: List[Dict[str, Any]]
    # Suggested skills
    suggested_skills: List[SkillSuggestion]
    # Metadata
    query: str
    observation_count: int
    skill_count: int
    total_tokens_estimate: int


class SkillsIntegration:
    """Integrates ai-skills with ai-mem for combined memory + expertise.

    Usage:
        integration = SkillsIntegration()
        if integration.available:
            skills = integration.find_relevant_skills("JWT authentication")
            enriched = await integration.search_with_skills(manager, "JWT auth")
    """

    def __init__(self):
        """Initialize skills integration."""
        self._router = None
        self._registry = None
        self._available = AISKILLS_AVAILABLE

    @property
    def available(self) -> bool:
        """Check if ai-skills is available."""
        return self._available

    def _get_router(self):
        """Lazily get the skill router."""
        if not self._available:
            return None
        if self._router is None:
            try:
                self._router = get_skill_router()
            except Exception as e:
                logger.warning(f"Failed to initialize skill router: {e}")
                self._available = False
        return self._router

    def find_relevant_skills(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.3,
    ) -> List[SkillSuggestion]:
        """Find skills relevant to a query.

        Args:
            query: Search query
            limit: Maximum number of skills to return
            min_score: Minimum relevance score (0-1)

        Returns:
            List of relevant skill suggestions
        """
        if not self._available:
            return []

        router = self._get_router()
        if not router:
            return []

        try:
            # Use router's browse method for lightweight search
            results = router.browse(context=query, limit=limit)

            suggestions = []
            for result in results:
                score = getattr(result, "score", 0.5)
                if score < min_score:
                    continue

                suggestions.append(
                    SkillSuggestion(
                        name=result.name,
                        description=getattr(result, "description", "")[:200],
                        category=getattr(result, "category", "general"),
                        relevance_score=score,
                        tokens_estimate=getattr(result, "tokens_est", 500),
                        tags=getattr(result, "tags", []),
                    )
                )

            return suggestions[:limit]

        except Exception as e:
            logger.warning(f"Error finding skills: {e}")
            return []

    def get_skill_content(self, skill_name: str, variables: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get the full content of a skill.

        Args:
            skill_name: Name of the skill
            variables: Optional variables to customize the skill

        Returns:
            Skill content or None if not found
        """
        if not self._available:
            return None

        router = self._get_router()
        if not router:
            return None

        try:
            result = router.use(context=skill_name, variables=variables, auto_select=True)
            if result and not result.ambiguous:
                return result.content
        except Exception as e:
            logger.warning(f"Error getting skill content: {e}")

        return None

    async def search_with_skills(
        self,
        manager,  # MemoryManager
        query: str,
        observation_limit: int = 5,
        skill_limit: int = 3,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> EnrichedSearchResult:
        """Search memory and find relevant skills in one call.

        This is the main integration method that combines:
        1. Memory search (observations from ai-mem)
        2. Skill discovery (expert knowledge from ai-skills)

        Args:
            manager: MemoryManager instance
            query: Search query
            observation_limit: Max observations to return
            skill_limit: Max skills to suggest
            project: Filter by project
            session_id: Filter by session

        Returns:
            EnrichedSearchResult with observations and skill suggestions
        """
        # Search observations
        observations = []
        try:
            search_results = await manager.search(
                query,
                limit=observation_limit,
                project=project,
                session_id=session_id,
            )
            if search_results:
                ids = [r.id for r in search_results]
                observations = await manager.get_observations(ids)
        except Exception as e:
            logger.warning(f"Error searching observations: {e}")

        # Find relevant skills
        skills = self.find_relevant_skills(query, limit=skill_limit)

        # Calculate total tokens
        obs_tokens = sum(
            len((obs.get("summary") or obs.get("content") or "").split()) * 1.3
            for obs in observations
        )
        skill_tokens = sum(s.tokens_estimate for s in skills)

        return EnrichedSearchResult(
            observations=observations,
            suggested_skills=skills,
            query=query,
            observation_count=len(observations),
            skill_count=len(skills),
            total_tokens_estimate=int(obs_tokens + skill_tokens),
        )

    def suggest_skills_for_observation(
        self,
        content: str,
        obs_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 2,
    ) -> List[SkillSuggestion]:
        """Suggest skills based on observation content.

        This can be used to automatically enrich observations with relevant expertise.

        Args:
            content: Observation content
            obs_type: Observation type
            tags: Observation tags
            limit: Max skills to suggest

        Returns:
            List of skill suggestions
        """
        if not self._available:
            return []

        # Build query from observation metadata
        query_parts = []

        if obs_type:
            # Map observation types to query hints
            type_hints = {
                "bugfix": "debugging error fix",
                "feature": "implementation new feature",
                "decision": "architecture design decision",
                "refactor": "refactoring code improvement",
                "tool_output": "tool usage",
            }
            if obs_type in type_hints:
                query_parts.append(type_hints[obs_type])

        if tags:
            query_parts.extend(tags[:3])

        # Add first 100 chars of content
        query_parts.append(content[:100])

        query = " ".join(query_parts)
        return self.find_relevant_skills(query, limit=limit, min_score=0.4)

    def format_skills_for_context(
        self,
        skills: List[SkillSuggestion],
        include_content: bool = False,
    ) -> str:
        """Format skill suggestions for inclusion in context.

        Args:
            skills: List of skill suggestions
            include_content: Whether to include full skill content

        Returns:
            Formatted string for context injection
        """
        if not skills:
            return ""

        lines = ["\n--- Relevant Expert Knowledge ---"]

        for skill in skills:
            lines.append(f"\n**{skill.name}** ({skill.category})")
            lines.append(f"  {skill.description}")
            if skill.tags:
                lines.append(f"  Tags: {', '.join(skill.tags[:5])}")

            if include_content:
                content = self.get_skill_content(skill.name)
                if content:
                    # Truncate to reasonable size
                    truncated = content[:1000] + ("..." if len(content) > 1000 else "")
                    lines.append(f"\n  Content:\n  {truncated}")

        lines.append("\n--- End Expert Knowledge ---")

        return "\n".join(lines)


# Singleton instance
_skills_integration: Optional[SkillsIntegration] = None


def get_skills_integration() -> SkillsIntegration:
    """Get or create the skills integration singleton.

    Returns:
        SkillsIntegration instance
    """
    global _skills_integration

    if _skills_integration is None:
        _skills_integration = SkillsIntegration()

    return _skills_integration


def is_skills_available() -> bool:
    """Check if ai-skills integration is available.

    Returns:
        True if ai-skills is installed and working
    """
    return get_skills_integration().available
