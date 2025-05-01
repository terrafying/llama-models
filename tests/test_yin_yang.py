import pytest
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import random
from datetime import datetime

@dataclass
class TestBranch:
    name: str
    tests: List[Dict[str, Any]]
    nature: str  # 'yin' or 'yang'
    fitness: float = 0.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class YinYangTestEcosystem:
    def __init__(self):
        self.branches: List[TestBranch] = []
        self.offspring_history: List[Tuple[TestBranch, TestBranch, TestBranch]] = []  # (parent_a, parent_b, offspring)

    def add_branch(self, branch: TestBranch):
        self.branches.append(branch)
        self._update_fitness(branch)

    def _update_fitness(self, branch: TestBranch):
        """Calculate fitness based on test coverage and success rate"""
        if not branch.tests:
            branch.fitness = 0.0
            return

        success_rate = sum(1 for test in branch.tests if test.get('passed', False)) / len(branch.tests)
        coverage = sum(1 for test in branch.tests if test.get('covered', False)) / len(branch.tests)
        branch.fitness = (success_rate + coverage) / 2

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

            # Mutate some properties
            if random.random() < 0.3:
                base_test['parameters'] = self._mutate_parameters(base_test.get('parameters', {}))

            combined_tests.append(base_test)

        # Create new branch with opposite nature of the more fit parent
        new_nature = 'yang' if parent_a.fitness > parent_b.fitness else 'yin'
        offspring = TestBranch(
            name=f"offspring_{len(self.offspring_history)}",
            tests=combined_tests,
            nature=new_nature
        )

        self.offspring_history.append((parent_a, parent_b, offspring))
        return offspring

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

@pytest.fixture
def yin_yang_ecosystem():
    return YinYangTestEcosystem()

def test_branch_creation(yin_yang_ecosystem):
    yin_branch = TestBranch(
        name="yin_branch",
        tests=[{'name': 'test1', 'passed': True, 'covered': True}],
        nature='yin'
    )
    yang_branch = TestBranch(
        name="yang_branch",
        tests=[{'name': 'test2', 'passed': True, 'covered': True}],
        nature='yang'
    )
    
    yin_yang_ecosystem.add_branch(yin_branch)
    yin_yang_ecosystem.add_branch(yang_branch)
    
    assert len(yin_yang_ecosystem.branches) == 2
    assert yin_branch.fitness > 0
    assert yang_branch.fitness > 0

def test_offspring_creation(yin_yang_ecosystem):
    yin_branch = TestBranch(
        name="yin_branch",
        tests=[{'name': 'test1', 'passed': True, 'covered': True}],
        nature='yin'
    )
    yang_branch = TestBranch(
        name="yang_branch",
        tests=[{'name': 'test2', 'passed': True, 'covered': True}],
        nature='yang'
    )
    
    yin_yang_ecosystem.add_branch(yin_branch)
    yin_yang_ecosystem.add_branch(yang_branch)
    
    offspring = yin_yang_ecosystem.create_offspring(yin_branch, yang_branch)
    
    assert offspring.nature in ['yin', 'yang']
    assert len(offspring.tests) > 0
    assert len(yin_yang_ecosystem.offspring_history) == 1

def test_compatible_branches(yin_yang_ecosystem):
    yin_branch = TestBranch(
        name="yin_branch",
        tests=[{'name': 'test1', 'passed': True, 'covered': True}],
        nature='yin'
    )
    yang_branch = TestBranch(
        name="yang_branch",
        tests=[{'name': 'test2', 'passed': True, 'covered': True}],
        nature='yang'
    )
    
    yin_yang_ecosystem.add_branch(yin_branch)
    yin_yang_ecosystem.add_branch(yang_branch)
    
    compatible = yin_yang_ecosystem.get_compatible_branches(yin_branch)
    assert len(compatible) == 1
    assert compatible[0].nature == 'yang' 