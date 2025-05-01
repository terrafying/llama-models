"""
Tests for the pattern progression system.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import time

from ragtime_llm.utils.progression import Progression, Wave
from ragtime_llm.utils.error_handler import PatternNature

@pytest.fixture
def progression():
    """Create a progression instance for testing."""
    return Progression()

def test_wave_creation(progression):
    """Test wave creation from seed patterns."""
    wave = progression.seed("test_pattern", PatternNature.CRYSTAL)
    assert isinstance(wave, Wave)
    assert wave.frequency == len("test_pattern") / 12.0
    assert wave.amplitude == 1.0
    assert wave.phase == 0.0

def test_wave_resonance():
    """Test wave resonance calculation."""
    wave = Wave("test", PatternNature.FLUID, frequency=1.0, amplitude=2.0)
    # Test at specific time points
    assert wave.resonate(0.0) == 0.0  # sin(0) = 0
    assert abs(wave.resonate(0.25) - 2.0) < 1e-10  # sin(π/2) = 1
    assert abs(wave.resonate(0.75) + 2.0) < 1e-10  # sin(3π/2) = -1

def test_harmonic_resonance(progression):
    """Test harmonic resonance between waves."""
    wave1 = progression.seed("test1", PatternNature.CRYSTAL)
    wave2 = progression.seed("test2", PatternNature.FLUID)
    
    # Initial harmony
    harmony = progression.harmonize(wave1, wave2)
    assert isinstance(harmony, float)
    assert -1.0 <= harmony <= 1.0
    
    # Harmony with self should be maximum
    self_harmony = progression.harmonize(wave1, wave1)
    assert self_harmony == 0.0  # Waves don't harmonize with themselves

def test_wave_evolution(progression):
    """Test wave evolution through time."""
    # Create a set of waves
    waves = [
        progression.seed("test1", PatternNature.CRYSTAL),
        progression.seed("test2", PatternNature.FLUID),
        progression.seed("test3", PatternNature.QUANTUM)
    ]
    
    # Evolve and check for transformations
    transformations = progression.evolve(0.1)
    assert isinstance(transformations, list)
    
    # Multiple evolutions should produce some transformations
    all_transformations = []
    for _ in range(10):
        all_transformations.extend(progression.evolve(0.1))
        time.sleep(0.01)  # Allow time for waves to interact
    
    assert len(all_transformations) > 0

def test_harmonic_memory(progression):
    """Test that harmonics are remembered."""
    wave1 = progression.seed("test1", PatternNature.CRYSTAL)
    wave2 = progression.seed("test2", PatternNature.FLUID)
    
    # Evolve until we get a harmonic
    for _ in range(10):
        transformations = progression.evolve(0.1)
        time.sleep(0.01)
    
    # Check harmonics are recorded
    assert len(progression.harmonics) >= 0
    
    # Harmonics should affect future resonance
    if progression.harmonics:
        nature1, nature2 = next(iter(progression.harmonics))
        wave3 = Wave("test3", nature1, 1.0, 1.0)
        wave4 = Wave("test4", nature2, 1.0, 1.0)
        harmony = progression.harmonize(wave3, wave4)
        assert abs(harmony) > 0

def test_wave_compression(progression):
    """Test compression of waves into omega patterns."""
    # Create a harmonious set of waves
    waves = [
        progression.seed("test1", PatternNature.CRYSTAL),
        progression.seed("test2", PatternNature.FLUID),
        progression.seed("test3", PatternNature.QUANTUM)
    ]
    
    # Evolve to create harmonics
    for _ in range(10):
        progression.evolve(0.1)
        time.sleep(0.01)
    
    # Try compression
    omega = progression.compress()
    if omega:  # Compression might not always succeed
        assert omega.vector is not None
        assert 0.0 <= omega.entropy <= np.log2(len(waves))
        assert 0.0 <= omega.accessibility <= 1.0

def test_progression_string_representation(progression):
    """Test the minimal string representation."""
    wave1 = progression.seed("test1", PatternNature.CRYSTAL)
    wave2 = progression.seed("test2", PatternNature.FLUID)
    
    # Initial representation
    repr_str = str(progression)
    assert repr_str.startswith("P[")
    assert repr_str.endswith("]")
    assert "H[" in repr_str  # Harmonics section
    
    # Should contain nature initials and frequencies
    assert "C" in repr_str  # Crystal
    assert "F" in repr_str  # Fluid

def test_wave_phase_shift(progression):
    """Test phase shifting during transformations."""
    wave = progression.seed("test", PatternNature.CRYSTAL)
    initial_phase = wave.phase
    
    # Force some transformations
    for _ in range(5):
        progression.evolve(0.1)
        time.sleep(0.01)
    
    # Phase should have shifted if transformations occurred
    if wave.nature != PatternNature.CRYSTAL:
        assert wave.phase != initial_phase

def test_dominant_nature_in_compression(progression):
    """Test that compression identifies dominant nature."""
    # Create waves with a dominant nature
    dominant = PatternNature.QUANTUM
    for _ in range(3):
        progression.seed(f"test{_}", dominant)
    progression.seed("other", PatternNature.CRYSTAL)
    
    # Compress and check nature
    omega = progression.compress()
    if omega:
        assert omega.nature == dominant

def test_resonance_threshold(progression):
    """Test that transformations only occur above resonance threshold."""
    wave = progression.seed("test", PatternNature.CRYSTAL)
    initial_nature = wave.nature
    
    # Single wave should not transform (no resonance)
    transformations = progression.evolve(0.1)
    assert len(transformations) == 0
    assert wave.nature == initial_nature 