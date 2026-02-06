"""Entity resolution — links raw entity mentions to canonical entities using fuzzy matching."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from rapidfuzz import fuzz, process
import json
from pathlib import Path

from src.utils import get_logger

logger = get_logger("entity_resolver")


@dataclass
class CanonicalEntity:
    """A canonical entity in the knowledge base."""
    canonical_id: str
    canonical_name: str
    entity_type: str  # person, project, topic, place, event, artifact
    aliases: List[str] = field(default_factory=list)
    first_mention: Optional[str] = None
    mention_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def add_alias(self, alias: str):
        """Add a new alias if not already present."""
        normalized = alias.lower().strip()
        if normalized not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)
            self.mention_count += 1


class EntityResolver:
    """Resolve entity mentions to canonical entities.

    Handles:
    - Name variations ("Bob" -> "Robert Chen")
    - Fuzzy matching for typos
    - Entity merging when same entity discovered
    - Persistence of entity state to disk
    """

    def __init__(self, fuzzy_threshold: float = 85.0, state_path: Optional[str] = None):
        self.fuzzy_threshold = fuzzy_threshold
        self.state_path = state_path
        self.canonical_entities: Dict[str, CanonicalEntity] = {}
        self.alias_index: Dict[str, str] = {}  # normalized alias -> canonical_id

        if state_path and Path(state_path).exists():
            self.load_state(state_path)
            logger.info(f"Loaded entity resolver state: {len(self.canonical_entities)} entities")

    def resolve(self, extracted_entities: List[Dict]) -> List[Dict]:
        """Resolve extracted entities to canonical entities.

        Args:
            extracted_entities: List of {name, type, relations} from EntityExtractor

        Returns:
            List of resolved entities with canonical_id added
        """
        resolved = []
        for entity in extracted_entities:
            mention = entity.get("name", "")
            if not mention:
                continue
            entity_type = entity.get("type", "unknown")

            # Step 1: Check exact alias match
            canonical_id = self._exact_match(mention)

            # Step 2: Try fuzzy matching
            if not canonical_id:
                canonical_id, confidence = self._fuzzy_match(mention, entity_type)
                if confidence < self.fuzzy_threshold:
                    canonical_id = None

            # Step 3: Create new canonical entity if no match
            if not canonical_id:
                canonical_id = self._create_canonical(mention, entity_type)
            else:
                self.canonical_entities[canonical_id].add_alias(mention)
                self._update_alias_index(mention, canonical_id)

            resolved.append({
                **entity,
                "canonical_id": canonical_id,
                "canonical_name": self.canonical_entities[canonical_id].canonical_name,
            })

        return resolved

    def _exact_match(self, mention: str) -> Optional[str]:
        """Check for exact alias match (case-insensitive)."""
        return self.alias_index.get(mention.lower().strip())

    def _fuzzy_match(self, mention: str, entity_type: str) -> Tuple[Optional[str], float]:
        """Find best fuzzy match among same-type entities."""
        all_candidates = []
        for cid, ce in self.canonical_entities.items():
            if ce.entity_type == entity_type or entity_type == "unknown":
                all_candidates.append((cid, ce.canonical_name))
                for alias in ce.aliases:
                    all_candidates.append((cid, alias))

        if not all_candidates:
            return None, 0.0

        names = [c[1] for c in all_candidates]
        result = process.extractOne(mention, names, scorer=fuzz.token_sort_ratio)

        if result:
            matched_name, score, idx = result
            return all_candidates[idx][0], score

        return None, 0.0

    def _create_canonical(self, name: str, entity_type: str) -> str:
        """Create a new canonical entity."""
        safe_name = name.lower().replace(" ", "_")[:20]
        canonical_id = f"{entity_type}_{len(self.canonical_entities)}_{safe_name}"

        self.canonical_entities[canonical_id] = CanonicalEntity(
            canonical_id=canonical_id,
            canonical_name=name,
            entity_type=entity_type,
            aliases=[name],
            first_mention=datetime.now().isoformat(),
            mention_count=1,
        )
        self._update_alias_index(name, canonical_id)
        return canonical_id

    def _update_alias_index(self, alias: str, canonical_id: str):
        self.alias_index[alias.lower().strip()] = canonical_id

    def merge_entities(self, source_id: str, target_id: str) -> bool:
        """Merge source entity into target entity."""
        if source_id not in self.canonical_entities or target_id not in self.canonical_entities:
            return False

        source = self.canonical_entities[source_id]
        target = self.canonical_entities[target_id]

        for alias in source.aliases:
            target.add_alias(alias)
            self._update_alias_index(alias, target_id)

        target.mention_count += source.mention_count
        target.metadata.update(source.metadata)
        del self.canonical_entities[source_id]
        return True

    def add_known_alias(self, canonical_id: str, alias: str) -> bool:
        """Manually add a known alias."""
        if canonical_id not in self.canonical_entities:
            return False
        self.canonical_entities[canonical_id].add_alias(alias)
        self._update_alias_index(alias, canonical_id)
        return True

    def get_all_entities(self) -> List[Dict]:
        """Return all canonical entities."""
        return [
            {
                "canonical_id": ce.canonical_id,
                "canonical_name": ce.canonical_name,
                "entity_type": ce.entity_type,
                "aliases": ce.aliases,
                "mention_count": ce.mention_count,
            }
            for ce in self.canonical_entities.values()
        ]

    def save_state(self, path: Optional[str] = None):
        """Save resolver state to disk."""
        path = path or self.state_path
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "canonical_entities": {
                cid: {
                    "canonical_id": ce.canonical_id,
                    "canonical_name": ce.canonical_name,
                    "entity_type": ce.entity_type,
                    "aliases": ce.aliases,
                    "first_mention": ce.first_mention,
                    "mention_count": ce.mention_count,
                    "metadata": ce.metadata,
                }
                for cid, ce in self.canonical_entities.items()
            },
            "alias_index": self.alias_index,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved entity resolver state ({len(self.canonical_entities)} entities)")

    def load_state(self, path: str):
        """Load resolver state from disk."""
        with open(path) as f:
            state = json.load(f)

        self.canonical_entities = {}
        for cid, data in state.get("canonical_entities", {}).items():
            self.canonical_entities[cid] = CanonicalEntity(
                canonical_id=data["canonical_id"],
                canonical_name=data["canonical_name"],
                entity_type=data["entity_type"],
                aliases=data.get("aliases", []),
                first_mention=data.get("first_mention"),
                mention_count=data.get("mention_count", 0),
                metadata=data.get("metadata", {}),
            )
        self.alias_index = state.get("alias_index", {})
