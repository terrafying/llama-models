"""
Tests for the error handler system, focusing on pattern evolution and omega patterns.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import re
import random

from ragtime_llm.utils.error_handler import (
    ErrorAnalyzer, ErrorContext, PatternContext, 
    VectorContext, OmegaPattern, PatternNature,
    PatternDynamics, PatternInteraction
)

@pytest.fixture
def error_analyzer():
    """Create an ErrorAnalyzer instance for testing."""
    return ErrorAnalyzer()

@pytest.fixture
def sample_error_context():
    """Create a sample error context for testing."""
    return ErrorContext(
        error_type="import_error",
        error_message="ImportError: No module named 'nonexistent_module'",
        file_path="test_file.py",
        line_number=10,
        test_name="test_import",
        module_name="nonexistent_module"
    )

def test_omega_pattern_creation(error_analyzer):
    """Test that omega patterns are created with correct initial properties."""
    pattern = "test_pattern"
    vector = np.random.rand(100)
    
    omega = error_analyzer._approximate_omega(pattern, vector)
    
    assert isinstance(omega, OmegaPattern)
    assert omega.attempts == 1
    assert omega.accessibility == 1.0
    assert omega.entropy > 0
    assert omega.last_approximation is not None
    assert len(omega.approximation_history) == 0

def test_omega_pattern_evolution(error_analyzer):
    """Test that omega patterns become more elusive with each attempt."""
    pattern = "test_pattern"
    vector = np.random.rand(100)
    
    # First approximation
    omega1 = error_analyzer._approximate_omega(pattern, vector)
    initial_accessibility = omega1.accessibility
    
    # Second approximation
    omega2 = error_analyzer._approximate_omega(pattern, vector)
    
    assert omega2.attempts == 2
    assert omega2.accessibility < initial_accessibility
    assert len(omega2.approximation_history) > 0

def test_vector_evolution_elusiveness(error_analyzer):
    """Test that vector evolution maintains elusiveness."""
    pattern = "test_pattern"
    initial_vector = np.random.rand(100)
    
    # Create vector context
    vector_context = VectorContext(
        vector=initial_vector,
        semantic_components=["test"],
        evolution_history=[]
    )
    
    # Evolve vector multiple times
    evolved_vectors = []
    for _ in range(5):
        new_vector = error_analyzer._evolve_vector(vector_context)
        evolved_vectors.append(new_vector)
        vector_context.vector = new_vector
        vector_context.evolution_history.append((new_vector, datetime.now()))
    
    # Check that vectors maintain some distance from each other
    for i in range(len(evolved_vectors) - 1):
        distance = np.linalg.norm(evolved_vectors[i] - evolved_vectors[i + 1])
        assert distance > 0.01  # Ensure some change between evolutions

def test_essential_form_uncertainty(error_analyzer):
    """Test that essential form finding maintains uncertainty."""
    pattern = "This is a test pattern with multiple words"
    vector = np.random.rand(100)
    
    # Get multiple essential forms
    forms = set()
    for _ in range(10):
        form = error_analyzer._find_essential_form(pattern, vector)
        forms.add(form)
    
    # Should not always return the same form
    assert len(forms) > 1

def test_pattern_context_evolution(error_analyzer, sample_error_context):
    """Test that pattern context evolves with increasing elusiveness."""
    pattern = "test_pattern"
    category = "dependency"
    error_type = "import_error"
    
    # Initial update
    error_analyzer._update_pattern_context(pattern, category, error_type)
    initial_context = error_analyzer.pattern_history[f"{category}:{error_type}"]
    initial_elusiveness = initial_context.elusiveness_score
    
    # Update multiple times
    for _ in range(5):
        error_analyzer._update_pattern_context(pattern, category, error_type)
    
    final_context = error_analyzer.pattern_history[f"{category}:{error_type}"]
    
    # Check that elusiveness increases
    assert final_context.elusiveness_score > initial_elusiveness
    assert final_context.omega_approximation is not None
    assert final_context.omega_approximation.attempts > 1

def test_compression_levels_with_elusiveness(error_analyzer):
    """Test that compression levels include elusiveness scores."""
    # Add some patterns
    patterns = [
        ("dependency:import_error", "import error pattern"),
        ("runtime:type_error", "type error pattern"),
        ("resource:memory_error", "memory error pattern")
    ]
    
    for pattern_key, pattern in patterns:
        error_analyzer._update_pattern_context(pattern, *pattern_key.split(":"))
    
    # Get compression levels
    levels = error_analyzer.get_compression_levels()
    
    # Check that each level includes elusiveness scores
    for level_patterns in levels.values():
        for pattern_key, essential_form, elusiveness in level_patterns:
            assert 0 <= elusiveness <= 1
            assert pattern_key in error_analyzer.pattern_history

def test_pattern_evolution_history(error_analyzer):
    """Test that pattern evolution history includes elusiveness scores."""
    pattern = "test_pattern"
    category = "test"
    error_type = "test_error"
    
    # Update pattern multiple times
    for _ in range(5):
        error_analyzer._update_pattern_context(pattern, category, error_type)
    
    # Get evolution history
    evolution = error_analyzer.get_pattern_evolution(f"{category}:{error_type}")
    
    # Check that each evolution step includes elusiveness
    for essential_form, timestamp, elusiveness in evolution:
        assert isinstance(essential_form, str)
        assert isinstance(timestamp, datetime)
        assert 0 <= elusiveness <= 1

def test_entropy_calculation(error_analyzer):
    """Test entropy calculation for vectors."""
    # Test with uniform vector
    uniform_vector = np.ones(100) / 100
    uniform_entropy = error_analyzer._calculate_entropy(uniform_vector)
    
    # Test with concentrated vector
    concentrated_vector = np.zeros(100)
    concentrated_vector[0] = 1
    concentrated_entropy = error_analyzer._calculate_entropy(concentrated_vector)
    
    # Uniform distribution should have higher entropy
    assert uniform_entropy > concentrated_entropy
    assert uniform_entropy > 0
    assert concentrated_entropy >= 0

def test_error_pattern_matching_with_elusiveness(error_analyzer):
    """Test error pattern matching with elusiveness consideration."""
    error_message = "ImportError: No module named 'test_module'"
    
    # Get initial match
    category, error_type, confidence, pattern_context = error_analyzer._match_error_pattern(error_message)
    
    # Match multiple times
    matches = []
    for _ in range(5):
        match = error_analyzer._match_error_pattern(error_message)
        matches.append(match)
    
    # Check that confidence varies due to elusiveness
    confidences = [match[2] for match in matches]
    assert len(set(confidences)) > 1  # Should not always be the same

def test_omega_pattern_approximation_history(error_analyzer):
    """Test that omega pattern approximation history is maintained."""
    pattern = "test_pattern"
    vector = np.random.rand(100)
    
    # Create multiple approximations
    for _ in range(10):
        omega = error_analyzer._approximate_omega(pattern, vector)
        vector = omega.last_approximation
    
    # Check history
    omega = error_analyzer.omega_patterns[pattern]
    assert len(omega.approximation_history) > 0
    
    # Check that each approximation has different entropy
    entropies = [entropy for _, entropy in omega.approximation_history]
    assert len(set(entropies)) > 1

def test_vector_context_omega_distance(error_analyzer):
    """Test that vector context maintains omega distance."""
    pattern = "test_pattern"
    category = "test"
    error_type = "test_error"
    
    # Initial update
    error_analyzer._update_pattern_context(pattern, category, error_type)
    initial_context = error_analyzer.vector_space[f"{category}:{error_type}"]
    initial_distance = initial_context.omega_distance
    
    # Update multiple times
    for _ in range(5):
        error_analyzer._update_pattern_context(pattern, category, error_type)
    
    final_context = error_analyzer.vector_space[f"{category}:{error_type}"]
    
    # Check that distance is maintained
    assert final_context.omega_distance != initial_distance
    assert final_context.omega_distance > 0
    assert final_context.approximation_confidence < 1.0

def test_pattern_nature_determination(error_analyzer):
    """Test that pattern natures are correctly determined."""
    # Test various error types
    assert error_analyzer._determine_pattern_nature("test", "syntax error") == PatternNature.CRYSTAL
    assert error_analyzer._determine_pattern_nature("test", "runtime error") == PatternNature.FLUID
    assert error_analyzer._determine_pattern_nature("test", "race condition") == PatternNature.QUANTUM
    assert error_analyzer._determine_pattern_nature("test", "system integration") == PatternNature.COSMIC
    assert error_analyzer._determine_pattern_nature("test", "import error") == PatternNature.VOID

def test_pattern_interaction_dynamics(error_analyzer):
    """Test pattern interaction dynamics."""
    # Test interaction strengths
    crystal_fluid = error_analyzer.pattern_dynamics.get_interaction(
        PatternNature.CRYSTAL, PatternNature.FLUID
    )
    assert crystal_fluid.influence > 0  # Crystal should beat Fluid
    
    quantum_crystal = error_analyzer.pattern_dynamics.get_interaction(
        PatternNature.QUANTUM, PatternNature.CRYSTAL
    )
    assert quantum_crystal.influence > 0  # Quantum should beat Crystal
    
    void_cosmic = error_analyzer.pattern_dynamics.get_interaction(
        PatternNature.VOID, PatternNature.COSMIC
    )
    assert void_cosmic.influence < 0  # Void should be weak to Cosmic

def test_pattern_nature_evolution(error_analyzer):
    """Test that pattern natures can evolve."""
    pattern_context = PatternContext(
        pattern="test_pattern",
        confidence=0.8,
        nature=PatternNature.CRYSTAL
    )
    
    # Evolve multiple times
    natures = set()
    for _ in range(20):
        new_nature = error_analyzer._evolve_pattern_nature(pattern_context)
        natures.add(new_nature)
        pattern_context.nature = new_nature
    
    # Should see multiple different natures
    assert len(natures) > 1

def test_omega_pattern_with_nature(error_analyzer):
    """Test omega pattern behavior with natures."""
    pattern = "test_pattern"
    vector = np.random.rand(100)
    nature = PatternNature.QUANTUM
    
    omega = error_analyzer._approximate_omega(pattern, vector, nature)
    assert omega.nature == nature
    assert len(omega.transformation_history) == 0
    
    # Multiple approximations should sometimes transform nature
    for _ in range(10):
        omega = error_analyzer._approximate_omega(pattern, vector, nature)
    
    assert len(omega.transformation_history) > 0

def test_vector_evolution_with_nature(error_analyzer):
    """Test vector evolution with nature influence."""
    pattern = "test_pattern"
    initial_vector = np.random.rand(100)
    
    # Create vector context with nature
    vector_context = VectorContext(
        vector=initial_vector,
        semantic_components=["test"],
        nature=PatternNature.FLUID,
        evolution_history=[]
    )
    
    # Evolve vector multiple times
    evolved_vectors = []
    natures = set()
    for _ in range(5):
        new_vector = error_analyzer._evolve_vector(vector_context)
        evolved_vectors.append(new_vector)
        vector_context.vector = new_vector
        vector_context.evolution_history.append((new_vector, datetime.now()))
        
        # Nature should influence evolution
        nature_influence = error_analyzer.pattern_dynamics.calculate_dominance(
            vector_context.nature,
            random.choice(list(PatternNature))
        )
        natures.add(nature_influence)
    
    # Check that nature influences create variation
    assert len(natures) > 1

def test_pattern_dominance_calculation(error_analyzer):
    """Test calculation of pattern dominance."""
    # Test each nature's dominance relationships
    for source in PatternNature:
        for target in PatternNature:
            if source != target:
                dominance = error_analyzer.pattern_dynamics.calculate_dominance(source, target)
                # Should have some non-zero influence
                assert abs(dominance) > 0
                # Should be within valid range
                assert -1 <= dominance <= 1

def test_pattern_transformation_chain(error_analyzer):
    """Test chain of pattern transformations."""
    pattern = "test_pattern"
    vector = np.random.rand(100)
    initial_nature = PatternNature.CRYSTAL
    
    # Create transformation chain
    transformations = []
    current_nature = initial_nature
    
    for _ in range(10):
        omega = error_analyzer._approximate_omega(pattern, vector, current_nature)
        if omega.nature != current_nature:
            transformations.append((current_nature, omega.nature))
            current_nature = omega.nature
    
    # Should see some transformations
    assert len(transformations) > 0
    # Should see some cycles or chains
    assert len(set(t[0] for t in transformations)) > 1

def test_nature_influence_on_entropy(error_analyzer):
    """Test how pattern nature influences entropy."""
    pattern = "test_pattern"
    vector = np.random.rand(100)
    
    # Test entropy with different natures
    entropies = {}
    for nature in PatternNature:
        omega = error_analyzer._approximate_omega(pattern, vector, nature)
        entropies[nature] = omega.entropy
    
    # Different natures should produce different entropy levels
    assert len(set(entropies.values())) > 1

def test_pattern_context_evolution(error_analyzer, sample_error_context):
    """Test pattern context evolution with natures."""
    pattern = "test_pattern"
    category = "dependency"
    error_type = "import_error"
    
    # Initial update
    error_analyzer._update_pattern_context(pattern, category, error_type)
    initial_context = error_analyzer.pattern_history[f"{category}:{error_type}"]
    initial_nature = initial_context.nature
    
    # Update multiple times
    natures_seen = {initial_nature}
    for _ in range(10):
        error_analyzer._update_pattern_context(pattern, category, error_type)
        current_context = error_analyzer.pattern_history[f"{category}:{error_type}"]
        natures_seen.add(current_context.nature)
    
    # Should see pattern nature evolution
    assert len(natures_seen) > 1
    assert current_context.omega_approximation is not None
    assert len(current_context.omega_approximation.transformation_history) > 0

def test_error_pattern_matching_with_nature(error_analyzer):
    """Test error pattern matching with nature consideration."""
    error_message = "ImportError: No module named 'test_module'"
    
    # Get initial match
    category, error_type, confidence, pattern_context = error_analyzer._match_error_pattern(error_message)
    initial_nature = pattern_context.nature
    
    # Match multiple times
    natures = {initial_nature}
    for _ in range(10):
        _, _, _, pattern_context = error_analyzer._match_error_pattern(error_message)
        natures.add(pattern_context.nature)
    
    # Should see nature evolution
    assert len(natures) > 1

def test_compression_with_nature_influence(error_analyzer):
    """Test that pattern compression is influenced by nature."""
    pattern = "This is a test pattern with multiple words for compression"
    vector = np.random.rand(100)
    
    # Test compression with different natures
    compressed_forms = {}
    for nature in PatternNature:
        vector_context = VectorContext(
            vector=vector,
            semantic_components=["test"],
            nature=nature,
            evolution_history=[]
        )
        error_analyzer._update_pattern_context(pattern, "test", "test_error")
        compressed = error_analyzer._find_essential_form(pattern, vector)
        compressed_forms[nature] = compressed
    
    # Different natures should produce different compressions
    assert len(set(compressed_forms.values())) > 1 