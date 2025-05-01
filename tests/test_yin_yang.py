import pytest
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
from datetime import datetime
import re
from collections import Counter

@dataclass
class TestBranch:
    name: str
    tests: List[Dict[str, Any]]
    nature: str  # 'yin' or 'yang'
    fitness: float = 0.0
    semantic_density: float = 0.0
    content_effectiveness: float = 0.0
    created_at: datetime = None
    parent_branches: Optional[Tuple['TestBranch', 'TestBranch']] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def calculate_semantic_density(self) -> float:
        """Calculate semantic density based on test names and descriptions"""
        if not self.tests:
            return 0.0
        
        # Extract meaningful words from test names and descriptions
        words = []
        for test in self.tests:
            words.extend(re.findall(r'\b\w+\b', test.get('name', '').lower()))
            if 'description' in test:
                words.extend(re.findall(r'\b\w+\b', test['description'].lower()))
        
        # Calculate semantic density based on unique meaningful words
        word_freq = Counter(words)
        unique_words = len(word_freq)
        total_words = len(words)
        
        return unique_words / total_words if total_words > 0 else 0.0

    def calculate_content_effectiveness(self) -> float:
        """Calculate content effectiveness based on test coverage and assertions"""
        if not self.tests:
            return 0.0
        
        effectiveness_scores = []
        for test in self.tests:
            # Score based on number of assertions
            assertions = len(test.get('assertions', []))
            # Score based on test complexity
            complexity = len(test.get('steps', []))
            # Score based on coverage
            coverage = 1.0 if test.get('covered', False) else 0.0
            
            effectiveness = (assertions * 0.4 + complexity * 0.3 + coverage * 0.3)
            effectiveness_scores.append(effectiveness)
        
        return sum(effectiveness_scores) / len(effectiveness_scores)

class YinYangTestEcosystem:
    def __init__(self):
        self.branches: List[TestBranch] = []
        self.offspring_history: List[Tuple[TestBranch, TestBranch, TestBranch]] = []
        self.min_semantic_density: float = 0.3
        self.min_content_effectiveness: float = 0.4

    def add_branch(self, branch: TestBranch):
        # Calculate metrics before adding
        branch.semantic_density = branch.calculate_semantic_density()
        branch.content_effectiveness = branch.calculate_content_effectiveness()
        
        # Only add if meets minimum requirements
        if (branch.semantic_density >= self.min_semantic_density and 
            branch.content_effectiveness >= self.min_content_effectiveness):
            self.branches.append(branch)
            self._update_fitness(branch)
        else:
            raise ValueError(
                f"Branch {branch.name} does not meet minimum requirements: "
                f"semantic_density={branch.semantic_density:.2f}, "
                f"content_effectiveness={branch.content_effectiveness:.2f}"
            )

    def _update_fitness(self, branch: TestBranch):
        """Calculate fitness based on multiple factors"""
        if not branch.tests:
            branch.fitness = 0.0
            return

        success_rate = sum(1 for test in branch.tests if test.get('passed', False)) / len(branch.tests)
        coverage = sum(1 for test in branch.tests if test.get('covered', False)) / len(branch.tests)
        
        # Weighted combination of all metrics
        branch.fitness = (
            success_rate * 0.3 +
            coverage * 0.3 +
            branch.semantic_density * 0.2 +
            branch.content_effectiveness * 0.2
        )

    def create_offspring(self, parent_a: TestBranch, parent_b: TestBranch) -> TestBranch:
        """Create a new test branch by combining tests from two parent branches"""
        if parent_a.nature == parent_b.nature:
            raise ValueError("Cannot create offspring from branches of the same nature")

        # Combine and mutate tests from both parents
        combined_tests = []
        for test_a, test_b in zip(parent_a.tests, parent_b.tests):
            # Randomly select and mutate test properties
            if random.random() < 0.5:
                base_test = test_a.copy()
            else:
                base_test = test_b.copy()

            # Enhance test with additional assertions and steps
            if 'assertions' not in base_test:
                base_test['assertions'] = []
            if 'steps' not in base_test:
                base_test['steps'] = []

            # Add complementary assertions from the other parent
            other_test = test_b if base_test == test_a else test_a
            if 'assertions' in other_test:
                base_test['assertions'].extend(other_test['assertions'])

            # Mutate parameters
            if random.random() < 0.3:
                base_test['parameters'] = self._mutate_parameters(base_test.get('parameters', {}))

            combined_tests.append(base_test)

        # Create new branch with opposite nature of the more fit parent
        new_nature = 'yang' if parent_a.fitness > parent_b.fitness else 'yin'
        offspring = TestBranch(
            name=f"offspring_{len(self.offspring_history)}",
            tests=combined_tests,
            nature=new_nature,
            parent_branches=(parent_a, parent_b)
        )

        # Calculate metrics for offspring
        offspring.semantic_density = offspring.calculate_semantic_density()
        offspring.content_effectiveness = offspring.calculate_content_effectiveness()

        # Only add to history if meets minimum requirements
        if (offspring.semantic_density >= self.min_semantic_density and 
            offspring.content_effectiveness >= self.min_content_effectiveness):
            self.offspring_history.append((parent_a, parent_b, offspring))
            return offspring
        else:
            raise ValueError(
                f"Generated offspring does not meet minimum requirements: "
                f"semantic_density={offspring.semantic_density:.2f}, "
                f"content_effectiveness={offspring.content_effectiveness:.2f}"
            )

    def _mutate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate test parameters to create variation"""
        mutated = params.copy()
        for key in mutated:
            if isinstance(mutated[key], (int, float)):
                mutated[key] *= random.uniform(0.8, 1.2)
            elif isinstance(mutated[key], str):
                if random.random() < 0.3:
                    mutated[key] = f"mutated_{mutated[key]}"
        return mutated

    def get_compatible_branches(self, branch: TestBranch) -> List[TestBranch]:
        """Get branches of opposite nature that can be used for creating offspring"""
        return [b for b in self.branches if b.nature != branch.nature]

    def get_best_branches(self, n: int = 2) -> List[TestBranch]:
        """Get the n best performing branches based on fitness"""
        return sorted(self.branches, key=lambda x: x.fitness, reverse=True)[:n]

@pytest.fixture
def yin_yang_ecosystem():
    return YinYangTestEcosystem()

def test_branch_creation(yin_yang_ecosystem):
    yin_branch = TestBranch(
        name="yin_branch",
        tests=[{
            'name': 'test1',
            'description': 'Test semantic density calculation',
            'passed': True,
            'covered': True,
            'assertions': ['assert_true', 'assert_equal'],
            'steps': ['step1', 'step2']
        }],
        nature='yin'
    )
    yang_branch = TestBranch(
        name="yang_branch",
        tests=[{
            'name': 'test2',
            'description': 'Test content effectiveness',
            'passed': True,
            'covered': True,
            'assertions': ['assert_not_none', 'assert_type'],
            'steps': ['step1', 'step2', 'step3']
        }],
        nature='yang'
    )
    
    yin_yang_ecosystem.add_branch(yin_branch)
    yin_yang_ecosystem.add_branch(yang_branch)
    
    assert len(yin_yang_ecosystem.branches) == 2
    assert yin_branch.fitness > 0
    assert yang_branch.fitness > 0
    assert yin_branch.semantic_density > 0
    assert yang_branch.content_effectiveness > 0

def test_offspring_creation(yin_yang_ecosystem):
    yin_branch = TestBranch(
        name="yin_branch",
        tests=[{
            'name': 'test1',
            'description': 'Test semantic density calculation',
            'passed': True,
            'covered': True,
            'assertions': ['assert_true', 'assert_equal'],
            'steps': ['step1', 'step2']
        }],
        nature='yin'
    )
    yang_branch = TestBranch(
        name="yang_branch",
        tests=[{
            'name': 'test2',
            'description': 'Test content effectiveness',
            'passed': True,
            'covered': True,
            'assertions': ['assert_not_none', 'assert_type'],
            'steps': ['step1', 'step2', 'step3']
        }],
        nature='yang'
    )
    
    yin_yang_ecosystem.add_branch(yin_branch)
    yin_yang_ecosystem.add_branch(yang_branch)
    
    offspring = yin_yang_ecosystem.create_offspring(yin_branch, yang_branch)
    
    assert offspring.nature in ['yin', 'yang']
    assert len(offspring.tests) > 0
    assert len(yin_yang_ecosystem.offspring_history) == 1
    assert offspring.semantic_density >= yin_yang_ecosystem.min_semantic_density
    assert offspring.content_effectiveness >= yin_yang_ecosystem.min_content_effectiveness
    assert offspring.parent_branches == (yin_branch, yang_branch)

def test_compatible_branches(yin_yang_ecosystem):
    yin_branch = TestBranch(
        name="yin_branch",
        tests=[{
            'name': 'test1',
            'description': 'Test semantic density calculation',
            'passed': True,
            'covered': True,
            'assertions': ['assert_true', 'assert_equal'],
            'steps': ['step1', 'step2']
        }],
        nature='yin'
    )
    yang_branch = TestBranch(
        name="yang_branch",
        tests=[{
            'name': 'test2',
            'description': 'Test content effectiveness',
            'passed': True,
            'covered': True,
            'assertions': ['assert_not_none', 'assert_type'],
            'steps': ['step1', 'step2', 'step3']
        }],
        nature='yang'
    )
    
    yin_yang_ecosystem.add_branch(yin_branch)
    yin_yang_ecosystem.add_branch(yang_branch)
    
    compatible = yin_yang_ecosystem.get_compatible_branches(yin_branch)
    assert len(compatible) == 1
    assert compatible[0].nature == 'yang' 