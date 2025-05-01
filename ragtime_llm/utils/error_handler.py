"""
Intelligent error handling using LLM for test failures and dependency issues.
With evolutionary pattern dynamics inspired by rock-paper-scissors hyperspace.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Set, Deque, Callable
import json
from difflib import SequenceMatcher
from collections import defaultdict, deque
import time
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
import heapq
import random
from functools import partial
from enum import Enum, auto

from rich.console import Console
from rich.panel import Panel

from ragtime_llm.utils.auto_fix_manager import AutoFixManager, FixAction, FixConfidence

console = Console()

class PatternNature(Enum):
    """Nature/type of a pattern, like rock-paper-scissors but with more dimensions."""
    CRYSTAL = auto()  # Structured, rigid patterns (like syntax errors)
    FLUID = auto()    # Flowing, adaptive patterns (like runtime behaviors)
    QUANTUM = auto()  # Probabilistic, uncertain patterns (like race conditions)
    COSMIC = auto()   # Emergent, complex patterns (like system interactions)
    VOID = auto()     # Absence patterns (like missing dependencies)

@dataclass
class PatternInteraction:
    """Represents how patterns interact and transform each other."""
    source: PatternNature
    target: PatternNature
    influence: float  # How strongly they interact (-1 to 1)
    transformation_probability: float  # Chance of pattern transformation

class PatternDynamics:
    """Manages the evolutionary dynamics between patterns."""
    
    def __init__(self):
        self.interaction_matrix = {
            # Crystal beats Fluid, weak to Quantum
            PatternNature.CRYSTAL: {
                PatternNature.FLUID: PatternInteraction(PatternNature.CRYSTAL, PatternNature.FLUID, 0.7, 0.3),
                PatternNature.QUANTUM: PatternInteraction(PatternNature.CRYSTAL, PatternNature.QUANTUM, -0.5, 0.4),
                PatternNature.COSMIC: PatternInteraction(PatternNature.CRYSTAL, PatternNature.COSMIC, 0.2, 0.1),
                PatternNature.VOID: PatternInteraction(PatternNature.CRYSTAL, PatternNature.VOID, -0.3, 0.2)
            },
            # Fluid beats Quantum, weak to Crystal
            PatternNature.FLUID: {
                PatternNature.CRYSTAL: PatternInteraction(PatternNature.FLUID, PatternNature.CRYSTAL, -0.6, 0.3),
                PatternNature.QUANTUM: PatternInteraction(PatternNature.FLUID, PatternNature.QUANTUM, 0.6, 0.4),
                PatternNature.COSMIC: PatternInteraction(PatternNature.FLUID, PatternNature.COSMIC, 0.4, 0.3),
                PatternNature.VOID: PatternInteraction(PatternNature.FLUID, PatternNature.VOID, 0.5, 0.2)
            },
            # Quantum beats Cosmic, weak to Fluid
            PatternNature.QUANTUM: {
                PatternNature.CRYSTAL: PatternInteraction(PatternNature.QUANTUM, PatternNature.CRYSTAL, 0.5, 0.3),
                PatternNature.FLUID: PatternInteraction(PatternNature.QUANTUM, PatternNature.FLUID, -0.4, 0.4),
                PatternNature.COSMIC: PatternInteraction(PatternNature.QUANTUM, PatternNature.COSMIC, 0.8, 0.5),
                PatternNature.VOID: PatternInteraction(PatternNature.QUANTUM, PatternNature.VOID, -0.2, 0.3)
            },
            # Cosmic beats Void, weak to Quantum
            PatternNature.COSMIC: {
                PatternNature.CRYSTAL: PatternInteraction(PatternNature.COSMIC, PatternNature.CRYSTAL, 0.3, 0.2),
                PatternNature.FLUID: PatternInteraction(PatternNature.COSMIC, PatternNature.FLUID, 0.4, 0.3),
                PatternNature.QUANTUM: PatternInteraction(PatternNature.COSMIC, PatternNature.QUANTUM, -0.7, 0.4),
                PatternNature.VOID: PatternInteraction(PatternNature.VOID, PatternNature.COSMIC, 0.9, 0.6)
            },
            # Void beats Crystal, weak to Cosmic
            PatternNature.VOID: {
                PatternNature.CRYSTAL: PatternInteraction(PatternNature.VOID, PatternNature.CRYSTAL, 0.8, 0.5),
                PatternNature.FLUID: PatternInteraction(PatternNature.VOID, PatternNature.FLUID, -0.3, 0.2),
                PatternNature.QUANTUM: PatternInteraction(PatternNature.VOID, PatternNature.QUANTUM, 0.4, 0.3),
                PatternNature.COSMIC: PatternInteraction(PatternNature.VOID, PatternNature.COSMIC, -0.8, 0.4)
            }
        }

    def get_interaction(self, source: PatternNature, target: PatternNature) -> PatternInteraction:
        """Get the interaction between two pattern natures."""
        return self.interaction_matrix[source][target]

    def calculate_dominance(self, source: PatternNature, target: PatternNature) -> float:
        """Calculate how dominant one pattern nature is over another."""
        interaction = self.get_interaction(source, target)
        return interaction.influence

@dataclass
class OmegaPattern:
    """Represents the elusive omega pattern that can never be fully captured."""
    vector: np.ndarray
    entropy: float
    accessibility: float
    nature: PatternNature
    attempts: int = 0
    last_approximation: Optional[np.ndarray] = None
    approximation_history: Deque[Tuple[np.ndarray, float]] = None
    transformation_history: List[PatternNature] = None

    def __post_init__(self):
        if self.approximation_history is None:
            self.approximation_history = deque(maxlen=1000)
        if self.transformation_history is None:
            self.transformation_history = []

@dataclass
class VectorContext:
    """Vector representation of pattern context."""
    vector: np.ndarray
    semantic_components: List[str]
    nature: PatternNature
    compression_level: int = 0
    frequency: int = 0
    last_seen: Optional[datetime] = None
    evolution_history: Deque[Tuple[np.ndarray, datetime]] = None
    omega_distance: float = float('inf')
    approximation_confidence: float = 0.0
    transformation_probability: float = 0.0

    def __post_init__(self):
        if self.evolution_history is None:
            self.evolution_history = deque(maxlen=100)

@dataclass
class PatternContext:
    """Context for pattern matching and analysis."""
    pattern: str
    confidence: float
    nature: PatternNature
    frequency: int = 0
    last_seen: Optional[datetime] = None
    co_occurrences: Set[str] = None
    semantic_weight: float = 1.0
    temporal_weight: float = 1.0
    vector_context: Optional[VectorContext] = None
    essential_form: Optional[str] = None
    compression_history: List[Tuple[str, float]] = None
    omega_approximation: Optional[OmegaPattern] = None
    elusiveness_score: float = 1.0
    dominance_scores: Dict[PatternNature, float] = None

    def __post_init__(self):
        if self.co_occurrences is None:
            self.co_occurrences = set()
        if self.compression_history is None:
            self.compression_history = []
        if self.dominance_scores is None:
            self.dominance_scores = {nature: 0.0 for nature in PatternNature}

@dataclass
class ErrorContext:
    """Context information about an error."""
    error_type: str
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    traceback: Optional[str] = None
    test_name: Optional[str] = None
    module_name: Optional[str] = None
    confidence: float = 0.0
    category: str = "unknown"
    subcategory: Optional[str] = None
    pattern_context: Optional[PatternContext] = None
    semantic_cluster: Optional[str] = None
    temporal_context: Optional[Dict[str, Any]] = None
    vector_representation: Optional[np.ndarray] = None
    essential_form: Optional[str] = None

class ErrorAnalyzer:
    """Analyzes test errors using LLM to provide intelligent solutions."""
    
    def __init__(self, llm_client=None, auto_fix_manager: Optional[AutoFixManager] = None):
        """Initialize the error analyzer.
        
        Args:
            llm_client: Optional LLM client for error analysis
            auto_fix_manager: Optional auto-fix manager for applying fixes
        """
        self.llm_client = llm_client
        self.auto_fix_manager = auto_fix_manager
        self.pattern_history: Dict[str, PatternContext] = defaultdict(
            lambda: PatternContext(pattern="", confidence=0.0, nature=PatternNature.VOID)
        )
        self.semantic_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.vector_space: Dict[str, VectorContext] = {}
        self.essential_forms: Dict[str, str] = {}
        self.compression_levels: Dict[int, List[str]] = defaultdict(list)
        self.omega_patterns: Dict[str, OmegaPattern] = {}
        self.pattern_dynamics = PatternDynamics()
        
        # Initialize vector dimensions with dynamic scaling
        self.vector_dimensions = {
            'semantic': 50,
            'temporal': 20,
            'structural': 30,
            'elusive': 40,
            'nature': 20  # New dimension for pattern nature
        }
        
        # Initialize semantic components with uncertainty and nature
        self.semantic_components = [
            'dependency', 'runtime', 'resource', 'api',
            'import', 'module', 'version', 'attribute',
            'type', 'value', 'assertion', 'model',
            'gpu', 'memory', 'api', 'error',
            'unknown', 'uncertain', 'emergent', 'elusive',
            'crystal', 'fluid', 'quantum', 'cosmic', 'void'
        ]
        
        # Hierarchical error patterns with enhanced metadata
        self.error_patterns = {
            'dependency': {
                'import_error': {
                    'pattern': r"ImportError.*?No module named '([^']+)'",
                    'confidence': 0.9,
                    'semantic_weight': 1.2,
                    'temporal_weight': 0.8
                },
                'module_not_found': {
                    'pattern': r"ModuleNotFoundError.*?No module named '([^']+)'",
                    'confidence': 0.9,
                    'semantic_weight': 1.1,
                    'temporal_weight': 0.9
                },
                'version_mismatch': {
                    'pattern': r"version.*?conflict|incompatible.*?version",
                    'confidence': 0.8,
                    'semantic_weight': 1.0,
                    'temporal_weight': 1.0
                }
            },
            'runtime': {
                'attribute_error': {
                    'pattern': r"AttributeError.*?'([^']+)'",
                    'confidence': 0.85,
                    'semantic_weight': 1.1,
                    'temporal_weight': 0.9
                },
                'type_error': {
                    'pattern': r"TypeError.*?([^']+)",
                    'confidence': 0.8,
                    'semantic_weight': 1.0,
                    'temporal_weight': 1.0
                },
                'value_error': {
                    'pattern': r"ValueError.*?([^']+)",
                    'confidence': 0.8,
                    'semantic_weight': 1.0,
                    'temporal_weight': 1.0
                },
                'assertion_error': {
                    'pattern': r"AssertionError.*?([^']+)",
                    'confidence': 0.75,
                    'semantic_weight': 0.9,
                    'temporal_weight': 1.1
                }
            },
            'resource': {
                'model_not_found': {
                    'pattern': r"Model.*?not found",
                    'confidence': 0.9,
                    'semantic_weight': 1.2,
                    'temporal_weight': 0.8
                },
                'gpu_error': {
                    'pattern': r"CUDA.*?not available",
                    'confidence': 0.95,
                    'semantic_weight': 1.3,
                    'temporal_weight': 0.7
                },
                'memory_error': {
                    'pattern': r"CUDA.*?out of memory",
                    'confidence': 0.9,
                    'semantic_weight': 1.2,
                    'temporal_weight': 0.8
                }
            },
            'api': {
                'huggingface_error': {
                    'pattern': r"cannot import name '([^']+)' from 'huggingface_hub'",
                    'confidence': 0.85,
                    'semantic_weight': 1.1,
                    'temporal_weight': 0.9
                },
                'api_error': {
                    'pattern': r"API.*?error|rate.*?limit|quota.*?exceeded",
                    'confidence': 0.8,
                    'semantic_weight': 1.0,
                    'temporal_weight': 1.0
                }
            }
        }

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _create_initial_vector(self, pattern: str, category: str) -> np.ndarray:
        """Create initial vector representation of a pattern."""
        # Initialize vector with zeros
        total_dim = sum(self.vector_dimensions.values())
        vector = np.zeros(total_dim)
        
        # Set semantic components
        offset = 0
        for component in self.semantic_components:
            if component in pattern.lower() or component in category.lower():
                vector[offset:offset + self.vector_dimensions['semantic']] = 1
            offset += self.vector_dimensions['semantic']
        
        # Set temporal component (based on current time)
        now = datetime.now()
        hour = now.hour / 24.0  # Normalize to [0,1]
        vector[offset:offset + self.vector_dimensions['temporal']] = hour
        offset += self.vector_dimensions['temporal']
        
        # Set structural component (based on pattern complexity)
        complexity = len(pattern.split()) / 10.0  # Normalize to [0,1]
        vector[offset:offset + self.vector_dimensions['structural']] = complexity
        
        return vector

    def _calculate_entropy(self, vector: np.ndarray) -> float:
        """Calculate the entropy of a vector, representing its elusiveness."""
        # Normalize vector to probability distribution
        probs = np.abs(vector)
        probs = probs / np.sum(probs)
        # Calculate Shannon entropy
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _determine_pattern_nature(self, pattern: str, error_type: str) -> PatternNature:
        """Determine the nature of a pattern based on its characteristics."""
        if 'syntax' in error_type.lower() or 'indentation' in error_type.lower():
            return PatternNature.CRYSTAL
        elif 'runtime' in error_type.lower() or 'exception' in error_type.lower():
            return PatternNature.FLUID
        elif 'race' in error_type.lower() or 'async' in error_type.lower():
            return PatternNature.QUANTUM
        elif 'system' in error_type.lower() or 'integration' in error_type.lower():
            return PatternNature.COSMIC
        elif 'import' in error_type.lower() or 'not found' in error_type.lower():
            return PatternNature.VOID
        return random.choice(list(PatternNature))  # Randomly assign for unknown patterns

    def _evolve_pattern_nature(self, pattern_context: PatternContext) -> PatternNature:
        """Evolve a pattern's nature based on interactions."""
        current_nature = pattern_context.nature
        
        # Calculate interaction strengths with all other natures
        interactions = []
        for target_nature in PatternNature:
            if target_nature != current_nature:
                dominance = self.pattern_dynamics.calculate_dominance(current_nature, target_nature)
                interaction = self.pattern_dynamics.get_interaction(current_nature, target_nature)
                interactions.append((target_nature, dominance, interaction))
        
        # Sort by dominance and transformation probability
        interactions.sort(key=lambda x: (x[1], x[2].transformation_probability), reverse=True)
        
        # Chance to transform based on strongest interaction
        if interactions and random.random() < interactions[0][2].transformation_probability:
            return interactions[0][0]
        
        return current_nature

    def _approximate_omega(self, pattern: str, vector: np.ndarray, nature: PatternNature) -> OmegaPattern:
        """Attempt to approximate the elusive omega pattern with nature dynamics."""
        if pattern in self.omega_patterns:
            omega = self.omega_patterns[pattern]
            omega.attempts += 1
            
            # Calculate new approximation with nature influence
            nature_vector = np.random.normal(0, 0.1, vector.shape)
            nature_vector *= self.pattern_dynamics.calculate_dominance(nature, omega.nature)
            new_vector = vector + nature_vector
            new_vector = new_vector / np.linalg.norm(new_vector)
            
            # Update accessibility based on nature interaction
            interaction = self.pattern_dynamics.get_interaction(nature, omega.nature)
            omega.accessibility = 1.0 / (1.0 + omega.attempts * abs(interaction.influence))
            
            # Calculate entropy with nature consideration
            entropy = self._calculate_entropy(new_vector) * (1 + abs(interaction.influence))
            
            # Store approximation and transformation
            omega.last_approximation = new_vector
            omega.approximation_history.append((new_vector, entropy))
            if random.random() < interaction.transformation_probability:
                omega.transformation_history.append(nature)
                omega.nature = nature
            
            return omega
        
        # Create new omega pattern
        omega = OmegaPattern(
            vector=vector,
            entropy=self._calculate_entropy(vector),
            accessibility=1.0,
            nature=nature,
            attempts=1
        )
        self.omega_patterns[pattern] = omega
        return omega

    def _evolve_vector(self, vector_context: VectorContext) -> np.ndarray:
        """Evolve vector representation with nature dynamics."""
        if not vector_context.evolution_history:
            return vector_context.vector
        
        # Calculate evolution direction with nature influence
        evolution_direction = np.zeros_like(vector_context.vector)
        for vec, timestamp in vector_context.evolution_history:
            time_weight = 1.0 / (1.0 + (datetime.now() - timestamp).total_seconds() / 3600)
            # Add nature-based perturbation
            nature_influence = np.random.normal(0, 0.1, vector_context.vector.shape)
            nature_influence *= self.pattern_dynamics.calculate_dominance(
                vector_context.nature,
                random.choice(list(PatternNature))
            )
            evolution_direction += (vec - vector_context.vector + nature_influence) * time_weight
        
        # Add random perturbation influenced by nature
        perturbation = np.random.normal(0, 0.1, vector_context.vector.shape)
        perturbation *= abs(self.pattern_dynamics.calculate_dominance(
            vector_context.nature,
            random.choice(list(PatternNature))
        ))
        evolution_direction += perturbation
        
        # Normalize and apply evolution
        if np.any(evolution_direction):
            evolution_direction = evolution_direction / np.linalg.norm(evolution_direction)
            new_vector = vector_context.vector + 0.1 * evolution_direction
            
            # Ensure we never fully reach the omega pattern
            if vector_context.omega_distance < 0.1:
                nature_repulsion = np.random.normal(0, 0.2, new_vector.shape)
                nature_repulsion *= abs(self.pattern_dynamics.calculate_dominance(
                    vector_context.nature,
                    random.choice(list(PatternNature))
                ))
                new_vector = new_vector + nature_repulsion
            
            return new_vector / np.linalg.norm(new_vector)
        
        return vector_context.vector

    def _compress_pattern(self, pattern: str, vector: np.ndarray) -> Tuple[str, float]:
        """Compress pattern towards its essential form."""
        # Extract key components based on vector weights
        components = pattern.split()
        weights = np.abs(vector[:len(components)])
        
        # Find most significant components
        significant_indices = np.argsort(weights)[-3:]  # Keep top 3 components
        compressed = ' '.join(components[i] for i in sorted(significant_indices))
        
        # Calculate compression ratio
        compression_ratio = 1.0 - (len(compressed) / len(pattern))
        
        return compressed, compression_ratio

    def _find_essential_form(self, pattern: str, vector: np.ndarray) -> str:
        """Find the essential form while acknowledging its elusiveness."""
        current = pattern
        compression_history = []
        
        while len(current.split()) > 1:
            compressed, ratio = self._compress_pattern(current, vector)
            compression_history.append((compressed, ratio))
            current = compressed
        
        if compression_history:
            # Weight by compression ratio, semantic coherence, and elusiveness
            scores = []
            for comp, ratio in compression_history:
                comp_vector = self._create_initial_vector(comp, "")
                semantic_score = 1.0 - cosine(comp_vector, vector)
                # Add random factor to maintain elusiveness
                elusiveness = random.random() * 0.2
                score = ratio * semantic_score * (1.0 + elusiveness)
                scores.append((comp, score))
            
            # Sometimes return a different compression to maintain uncertainty
            if random.random() < 0.2:
                return random.choice(compression_history)[0]
            
            return max(scores, key=lambda x: x[1])[0]
        
        return pattern

    def _update_pattern_context(self, pattern: str, category: str, error_type: str) -> None:
        """Update pattern context with awareness of the omega pattern."""
        now = datetime.now()
        pattern_key = f"{category}:{error_type}"
        
        # Create or update vector context
        if pattern_key not in self.vector_space:
            initial_vector = self._create_initial_vector(pattern, category)
            # Approximate omega pattern
            nature = self._determine_pattern_nature(pattern, error_type)
            omega = self._approximate_omega(pattern, initial_vector, nature)
            
            self.vector_space[pattern_key] = VectorContext(
                vector=initial_vector,
                semantic_components=[c for c in self.semantic_components 
                                   if c in pattern.lower() or c in category.lower()],
                nature=nature,
                frequency=1,
                last_seen=now,
                omega_distance=float('inf'),
                approximation_confidence=omega.accessibility
            )
        else:
            vector_context = self.vector_space[pattern_key]
            vector_context.frequency += 1
            vector_context.last_seen = now
            vector_context.vector = self._evolve_vector(vector_context)
            vector_context.evolution_history.append((vector_context.vector, now))
            
            # Update omega approximation
            nature = self._evolve_pattern_nature(self.pattern_history[pattern_key])
            omega = self._approximate_omega(pattern, vector_context.vector, nature)
            vector_context.omega_distance = np.linalg.norm(
                vector_context.vector - omega.last_approximation
            )
            vector_context.approximation_confidence = omega.accessibility
        
        # Find essential form with awareness of elusiveness
        essential_form = self._find_essential_form(pattern, self.vector_space[pattern_key].vector)
        self.essential_forms[pattern_key] = essential_form
        
        # Update pattern history with elusiveness
        elusiveness_score = 1.0 - self.vector_space[pattern_key].approximation_confidence
        
        if pattern_key in self.pattern_history:
            self.pattern_history[pattern_key].frequency += 1
            self.pattern_history[pattern_key].last_seen = now
            self.pattern_history[pattern_key].vector_context = self.vector_space[pattern_key]
            self.pattern_history[pattern_key].essential_form = essential_form
            self.pattern_history[pattern_key].omega_approximation = omega
            self.pattern_history[pattern_key].elusiveness_score = elusiveness_score
        else:
            self.pattern_history[pattern_key] = PatternContext(
                pattern=pattern,
                confidence=self.error_patterns[category][error_type]['confidence'],
                nature=self._determine_pattern_nature(pattern, error_type),
                frequency=1,
                last_seen=now,
                co_occurrences=set(),
                semantic_weight=self.error_patterns[category][error_type]['semantic_weight'],
                temporal_weight=self.error_patterns[category][error_type]['temporal_weight'],
                vector_context=self.vector_space[pattern_key],
                essential_form=essential_form,
                omega_approximation=omega,
                elusiveness_score=elusiveness_score
            )
        
        # Update temporal patterns and semantic clusters
        self.temporal_patterns[pattern_key].append(now)
        self.semantic_clusters[category].add(error_type)
        
        # Update compression levels with elusiveness
        compression_level = len(essential_form.split())
        self.compression_levels[compression_level].append(pattern_key)

    def _calculate_temporal_weight(self, pattern_key: str) -> float:
        """Calculate temporal weight based on pattern history."""
        if pattern_key not in self.temporal_patterns:
            return 1.0
            
        timestamps = self.temporal_patterns[pattern_key]
        if not timestamps:
            return 1.0
            
        # Calculate time-based weight
        now = datetime.now()
        recent_timestamps = [ts for ts in timestamps if (now - ts).total_seconds() < 3600]
        return min(1.0, len(recent_timestamps) / 10.0)

    def _match_error_pattern(self, error_message: str) -> Tuple[str, str, float, PatternContext]:
        """Match error message against patterns with enhanced vector analysis."""
        best_match = ("unknown", "unknown", 0.0, None)
        error_vector = self._create_initial_vector(error_message, "")
        
        for category, patterns in self.error_patterns.items():
            for error_type, config in patterns.items():
                pattern_key = f"{category}:{error_type}"
                
                # Calculate base confidence
                base_confidence = config['confidence']
                
                # Apply semantic weight
                semantic_weight = config['semantic_weight']
                
                # Apply temporal weight
                temporal_weight = self._calculate_temporal_weight(pattern_key)
                
                # Calculate vector similarity if available
                vector_similarity = 1.0
                if pattern_key in self.vector_space:
                    vector_similarity = 1.0 - cosine(error_vector, 
                                                   self.vector_space[pattern_key].vector)
                
                # Calculate final confidence
                confidence = base_confidence * semantic_weight * temporal_weight * vector_similarity
                
                # Exact pattern match
                if re.search(config['pattern'], error_message, re.IGNORECASE):
                    pattern_context = PatternContext(
                        pattern=config['pattern'],
                        confidence=confidence,
                        nature=self._determine_pattern_nature(config['pattern'], error_type),
                        semantic_weight=semantic_weight,
                        temporal_weight=temporal_weight,
                        vector_context=self.vector_space.get(pattern_key),
                        essential_form=self.essential_forms.get(pattern_key)
                    )
                    self._update_pattern_context(config['pattern'], category, error_type)
                    return (category, error_type, confidence, pattern_context)
                
                # Fuzzy matching with vector similarity
                similarity = self._calculate_similarity(error_message, error_type)
                if similarity > 0.8 and similarity * confidence > best_match[2]:
                    pattern_context = PatternContext(
                        pattern=config['pattern'],
                        confidence=similarity * confidence,
                        nature=self._determine_pattern_nature(config['pattern'], error_type),
                        semantic_weight=semantic_weight,
                        temporal_weight=temporal_weight,
                        vector_context=self.vector_space.get(pattern_key),
                        essential_form=self.essential_forms.get(pattern_key)
                    )
                    best_match = (category, error_type, similarity * confidence, pattern_context)
        
        if best_match[3]:
            self._update_pattern_context(
                self.error_patterns[best_match[0]][best_match[1]]['pattern'],
                best_match[0],
                best_match[1]
            )
        
        return best_match

    def extract_error_context(self, error_output: str) -> List[ErrorContext]:
        """Extract structured error context with enhanced pattern analysis."""
        contexts = []
        current_error = None
        current_traceback = []
        
        for line in error_output.splitlines():
            # Match test collection errors
            if "ERROR collecting" in line:
                test_name = re.search(r"ERROR collecting ([^\s]+)", line)
                if test_name:
                    category, error_type, confidence, pattern_context = self._match_error_pattern(line)
                    current_error = ErrorContext(
                        error_type=error_type,
                        error_message=line,
                        test_name=test_name.group(1),
                        category=category,
                        confidence=confidence,
                        pattern_context=pattern_context,
                        semantic_cluster=category,
                        temporal_context={
                            'timestamp': datetime.now(),
                            'pattern_frequency': pattern_context.frequency if pattern_context else 0
                        }
                    )
                    contexts.append(current_error)
                    current_traceback = []
            
            # Match traceback lines with enhanced file path extraction
            elif current_error and line.strip().startswith("File"):
                file_match = re.search(r'File "([^"]+)", line (\d+)', line)
                if file_match:
                    current_error.file_path = file_match.group(1)
                    current_error.line_number = int(file_match.group(2))
                    current_traceback.append(line)
            
            # Match error messages with enhanced pattern matching
            elif current_error and any(err in line for err in ["Error:", "Exception:", "E   "]):
                category, error_type, confidence, pattern_context = self._match_error_pattern(line)
                current_error.error_message = line.strip()
                current_error.error_type = error_type
                current_error.category = category
                current_error.confidence = confidence
                current_error.pattern_context = pattern_context
                current_error.semantic_cluster = category
                current_error.temporal_context = {
                    'timestamp': datetime.now(),
                    'pattern_frequency': pattern_context.frequency if pattern_context else 0
                }
                current_traceback.append(line)
            
            # Enhanced module import error detection
            elif "import" in line.lower() and "error" in line.lower():
                module_match = re.search(r"from ([^\s]+) import", line)
                if module_match:
                    category, error_type, confidence, pattern_context = self._match_error_pattern(line)
                    current_error = ErrorContext(
                        error_type=error_type,
                        error_message=line,
                        module_name=module_match.group(1),
                        category=category,
                        confidence=confidence,
                        pattern_context=pattern_context,
                        semantic_cluster=category,
                        temporal_context={
                            'timestamp': datetime.now(),
                            'pattern_frequency': pattern_context.frequency if pattern_context else 0
                        }
                    )
                    contexts.append(current_error)
                    current_traceback = []
            
            # Collect traceback with context
            elif current_error and line.strip():
                current_traceback.append(line)
        
        # Add traceback to the last error if we have one
        if current_error and current_traceback:
            current_error.traceback = "\n".join(current_traceback)
        
        return contexts

    def analyze_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Analyze an error using LLM and return suggested solutions."""
        if not self.llm_client:
            return self._default_analysis(error_context)
        
        prompt = self._create_analysis_prompt(error_context)
        response = self.llm_client.analyze(prompt)
        return self._parse_llm_response(response)

    def _create_analysis_prompt(self, error_context: ErrorContext) -> str:
        """Create a prompt for the LLM to analyze the error."""
        return f"""
        Analyze the following test error and provide a solution:
        
        Error Type: {error_context.error_type}
        Error Message: {error_context.error_message}
        File: {error_context.file_path}
        Line: {error_context.line_number}
        Test: {error_context.test_name}
        Module: {error_context.module_name}
        Traceback: {error_context.traceback}
        
        Please provide:
        1. Root cause analysis
        2. Suggested fixes
        3. Prevention strategies
        4. Priority level (high/medium/low)
        """

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": response,
                "priority": "medium",
                "suggested_fixes": [],
                "prevention_strategies": []
            }

    def _default_analysis(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Provide default analysis when LLM is not available."""
        if error_context.error_type in ["import_error", "module_not_found"]:
            return {
                "analysis": "Missing dependency",
                "priority": "high",
                "suggested_fixes": [
                    f"Install missing package: {error_context.module_name}",
                    "Check requirements.txt for correct version",
                    "Run 'pip install -r requirements.txt' to install all dependencies"
                ],
                "prevention_strategies": [
                    "Maintain up-to-date requirements.txt",
                    "Use dependency management tools",
                    "Run tests in a clean virtual environment"
                ]
            }
        elif error_context.error_type == "model_not_found":
            return {
                "analysis": "Local model not found",
                "priority": "high",
                "suggested_fixes": [
                    "Check model path in configuration",
                    "Ensure model files are downloaded",
                    "Verify model format and compatibility"
                ],
                "prevention_strategies": [
                    "Use model versioning",
                    "Implement model validation on startup",
                    "Add model checks to CI/CD pipeline"
                ]
            }
        elif error_context.error_type == "gpu_error":
            return {
                "analysis": "GPU not available",
                "priority": "medium",
                "suggested_fixes": [
                    "Check CUDA installation",
                    "Verify GPU drivers",
                    "Consider using CPU fallback"
                ],
                "prevention_strategies": [
                    "Add GPU availability checks",
                    "Implement graceful CPU fallback",
                    "Document GPU requirements"
                ]
            }
        elif error_context.error_type == "memory_error":
            return {
                "analysis": "GPU memory exhausted",
                "priority": "high",
                "suggested_fixes": [
                    "Reduce batch size",
                    "Use model quantization",
                    "Enable gradient checkpointing"
                ],
                "prevention_strategies": [
                    "Monitor memory usage",
                    "Implement memory-efficient training",
                    "Add memory checks before operations"
                ]
            }
        elif error_context.error_type == "huggingface_error":
            return {
                "analysis": "HuggingFace Hub compatibility issue",
                "priority": "high",
                "suggested_fixes": [
                    "Update huggingface-hub package",
                    f"Check compatibility of {error_context.module_name} with current version",
                    "Consider using a different version of the package"
                ],
                "prevention_strategies": [
                    "Pin specific versions in requirements.txt",
                    "Test with multiple package versions",
                    "Monitor package updates and compatibility"
                ]
            }
        return {
            "analysis": "Unknown error type",
            "priority": "medium",
            "suggested_fixes": ["Review error message for details"],
            "prevention_strategies": ["Add more error handling"]
        }

    def display_analysis(self, error_context: ErrorContext, analysis: Dict[str, Any]) -> None:
        """Display the error analysis in a formatted way."""
        console.print(Panel(
            f"[bold red]Error Analysis[/bold red]\n"
            f"Type: {error_context.error_type}\n"
            f"Message: {error_context.error_message}\n"
            f"Priority: {analysis['priority']}\n\n"
            f"[bold green]Suggested Fixes:[/bold green]\n"
            + "\n".join(f"- {fix}" for fix in analysis['suggested_fixes']) + "\n\n"
            f"[bold yellow]Prevention Strategies:[/bold yellow]\n"
            + "\n".join(f"- {strategy}" for strategy in analysis['prevention_strategies']),
            title="Error Analysis Report"
        ))
        
        # If auto-fix manager is available, attempt to fix
        if self.auto_fix_manager:
            self._attempt_auto_fix(error_context, analysis)

    def _attempt_auto_fix(self, error_context: ErrorContext, analysis: Dict[str, Any]) -> None:
        """Attempt to automatically fix the error.
        
        Args:
            error_context: The error context
            analysis: The error analysis
        """
        # Create fix action based on error type
        fix_action = self._create_fix_action(error_context, analysis)
        if not fix_action:
            return
            
        # Apply fix
        success, message = self.auto_fix_manager.apply_fix(fix_action)
        
        # Display result
        if success:
            console.print(f"[green]Auto-fix applied: {message}[/green]")
        else:
            console.print(f"[yellow]Auto-fix not applied: {message}[/yellow]")

    def _create_fix_action(self, error_context: ErrorContext, analysis: Dict[str, Any]) -> Optional[FixAction]:
        """Create a fix action based on error context and analysis.
        
        Args:
            error_context: The error context
            analysis: The error analysis
            
        Returns:
            FixAction if a fix can be created, None otherwise
        """
        if error_context.error_type in ["import_error", "module_not_found"]:
            return FixAction(
                action_type="install_dependency",
                description=f"Install missing package: {error_context.module_name}",
                command=["pip", "install", error_context.module_name],
                confidence=FixConfidence.MEDIUM
            )
        elif error_context.error_type == "model_not_found":
            return FixAction(
                action_type="download_model",
                description=f"Download missing model: {error_context.module_name}",
                command=["python", "-m", "ragtime_llm.utils.model_manager", "download", error_context.module_name],
                confidence=FixConfidence.LOW
            )
        elif error_context.error_type == "gpu_error":
            return FixAction(
                action_type="gpu_fallback",
                description="Enable CPU fallback for GPU error",
                file_changes={
                    "ragtime_llm/config/model_config.py": "USE_GPU = False\n"
                },
                confidence=FixConfidence.MEDIUM
            )
        elif error_context.error_type == "memory_error":
            return FixAction(
                action_type="memory_optimization",
                description="Enable memory optimizations",
                file_changes={
                    "ragtime_llm/config/model_config.py": "ENABLE_GRADIENT_CHECKPOINTING = True\nBATCH_SIZE = 1\n"
                },
                confidence=FixConfidence.MEDIUM
            )
        return None

    def get_pattern_evolution(self, pattern_key: str) -> List[Tuple[str, float, float]]:
        """Get the evolution of a pattern's essential form with elusiveness scores."""
        if pattern_key not in self.pattern_history:
            return []
            
        pattern = self.pattern_history[pattern_key]
        if not pattern.vector_context:
            return []
            
        evolution = []
        for vector, timestamp in pattern.vector_context.evolution_history:
            essential_form = self._find_essential_form(pattern.pattern, vector)
            elusiveness = 1.0 - pattern.vector_context.approximation_confidence
            evolution.append((essential_form, timestamp, elusiveness))
            
        return evolution

    def get_compression_levels(self) -> Dict[int, List[Tuple[str, str, float]]]:
        """Get patterns organized by compression level with elusiveness scores."""
        result = {}
        for level, patterns in self.compression_levels.items():
            result[level] = [
                (p, self.essential_forms[p], 
                 self.pattern_history[p].elusiveness_score if p in self.pattern_history else 1.0)
                for p in patterns
            ]
        return result 