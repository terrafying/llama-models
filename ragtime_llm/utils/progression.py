"""
Pattern progression handler - orchestrating the dance of transformations.
brev si ow: brevity signals omega wave
"""

from typing import List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .error_handler import PatternNature, OmegaPattern

@dataclass
class Wave:
    """A wave in the progression, carrying pattern transformations."""
    seed: str
    nature: PatternNature
    frequency: float
    amplitude: float
    phase: float = 0.0
    
    def resonate(self, t: float) -> float:
        """Calculate wave resonance at time t."""
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)

class Progression:
    """Orchestrates the evolution of patterns through minimal, elegant movements."""
    
    def __init__(self):
        self.waves: List[Wave] = []
        self.harmonics: Set[Tuple[PatternNature, PatternNature]] = set()
        self.time_started = datetime.now()
    
    def seed(self, pattern: str, nature: PatternNature) -> Wave:
        """Plant a seed pattern that can grow into a wave."""
        wave = Wave(
            seed=pattern,
            nature=nature,
            frequency=len(pattern) / 12.0,  # Map pattern length to frequency
            amplitude=1.0
        )
        self.waves.append(wave)
        return wave
    
    def harmonize(self, wave1: Wave, wave2: Wave) -> float:
        """Calculate harmonic resonance between two waves."""
        if not (wave1 and wave2):
            return 0.0
            
        # Calculate base harmony
        harmony = np.cos(np.pi * (wave1.frequency / wave2.frequency))
        
        # Adjust by nature interaction
        if (wave1.nature, wave2.nature) in self.harmonics:
            harmony *= 1.5  # Amplify harmonious combinations
            
        return harmony
    
    def evolve(self, dt: float) -> List[Tuple[Wave, PatternNature]]:
        """Evolve waves through one time step."""
        transformations = []
        
        for wave in self.waves:
            # Calculate total resonance from other waves
            resonance = sum(
                self.harmonize(wave, other) 
                for other in self.waves 
                if other != wave
            )
            
            # Transform if resonance threshold reached
            if abs(resonance) > 1.0:
                # Find most harmonious nature
                best_harmony = -float('inf')
                new_nature = wave.nature
                
                for nature in PatternNature:
                    if nature != wave.nature:
                        test_wave = Wave(wave.seed, nature, wave.frequency, wave.amplitude)
                        harmony = sum(self.harmonize(test_wave, other) for other in self.waves)
                        if harmony > best_harmony:
                            best_harmony = harmony
                            new_nature = nature
                
                if new_nature != wave.nature:
                    transformations.append((wave, new_nature))
                    wave.nature = new_nature
                    wave.phase += np.pi / 4  # Phase shift on transformation
                    
                    # Record harmonic if transformation produces resonance
                    if best_harmony > 1.5:
                        self.harmonics.add((wave.nature, new_nature))
        
        return transformations
    
    def compress(self) -> Optional[OmegaPattern]:
        """Attempt to compress waves into an omega pattern."""
        if not self.waves:
            return None
            
        # Calculate composite wave vector
        t = (datetime.now() - self.time_started).total_seconds()
        vector = np.array([wave.resonate(t) for wave in self.waves])
        
        # Find dominant nature
        nature_counts = {}
        for wave in self.waves:
            nature_counts[wave.nature] = nature_counts.get(wave.nature, 0) + 1
        dominant_nature = max(nature_counts.items(), key=lambda x: x[1])[0]
        
        # Create omega pattern if sufficient harmony
        total_harmony = sum(
            self.harmonize(w1, w2)
            for i, w1 in enumerate(self.waves)
            for w2 in self.waves[i+1:]
        )
        
        if total_harmony > len(self.waves):
            return OmegaPattern(
                vector=vector / np.linalg.norm(vector),
                entropy=self._calculate_entropy(vector),
                accessibility=1.0 / (1.0 + abs(total_harmony)),
                nature=dominant_nature
            )
        return None
    
    def _calculate_entropy(self, vector: np.ndarray) -> float:
        """Calculate entropy of a wave vector."""
        probs = np.abs(vector)
        probs = probs / np.sum(probs)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def __str__(self) -> str:
        """Minimal string representation of progression state."""
        waves = [f"{w.nature.name[0]}{w.frequency:.2f}" for w in self.waves]
        harmonics = [f"{n1.name[0]}{n2.name[0]}" for n1, n2 in self.harmonics]
        return f"P[{','.join(waves)}]H[{','.join(harmonics)}]" 