"""
Comprehensive test suite for video generation with edge cases and content types.
"""

import os
import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
from ragtime_llm.core.unified_rag_system import YouTubeRAG
from ragtime_llm.video.video_generator import VideoGenerator
from ragtime_llm.video.video_composer import VideoComposer, VideoSegment

class VideoGenerationTester:
    def __init__(self, output_dir: str = "test_output"):
        """Initialize the tester.
        
        Args:
            output_dir: Directory to save test outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize RAG system
        self.rag_system = YouTubeRAG()
        
        # Test cases
        self.test_cases = [
            # Edge Case 1: Very short response
            {
                "name": "short_response",
                "query": "What is 2+2?",
                "expected_intensity": 0.3,
                "expected_effects": ["subtle_pulse"]
            },
            
            # Edge Case 2: Very long response
            {
                "name": "long_response",
                "query": "Explain the entire history of quantum physics in detail.",
                "expected_intensity": 0.7,
                "expected_effects": ["pulse", "glow"]
            },
            
            # Edge Case 3: High emotional content
            {
                "name": "emotional_content",
                "query": "What is the meaning of life and death?",
                "expected_intensity": 0.8,
                "expected_effects": ["pulse", "glow", "shake"]
            },
            
            # Edge Case 4: Technical content
            {
                "name": "technical_content",
                "query": "Explain the mathematical proof of Fermat's Last Theorem.",
                "expected_intensity": 0.5,
                "expected_effects": ["subtle_pulse"]
            },
            
            # Edge Case 5: Multiple speakers
            {
                "name": "multiple_speakers",
                "query": "Debate: Is artificial intelligence a threat to humanity?",
                "expected_intensity": 0.6,
                "expected_effects": ["pulse", "soft_glow"]
            },
            
            # Edge Case 6: Mixed content types
            {
                "name": "mixed_content",
                "query": "Explain both the technical and philosophical aspects of quantum entanglement.",
                "expected_intensity": 0.6,
                "expected_effects": ["pulse", "soft_glow"]
            }
        ]
        
    def run_all_tests(self):
        """Run all test cases and generate a report."""
        results = []
        
        for test_case in self.test_cases:
            print(f"\nRunning test: {test_case['name']}")
            result = self._run_test_case(test_case)
            results.append(result)
            
        self._generate_report(results)
        
    def _run_test_case(self, test_case: dict) -> dict:
        """Run a single test case.
        
        Args:
            test_case: Test case configuration
            
        Returns:
            Test result data
        """
        try:
            # Generate video response
            output_path = os.path.join(self.output_dir, f"{test_case['name']}.mp4")
            result = self.rag_system.generate_video_response(
                query=test_case["query"],
                voice_id="JOE_ROGAN_VOICE_ID",
                output_dir=self.output_dir,
                intensity=test_case["expected_intensity"]
            )
            
            # Load composition data
            composition_path = result.replace(".mp4", ".json")
            with open(composition_path, 'r') as f:
                composition = json.load(f)
                
            # Verify results
            actual_intensity = composition.get("intensity", 0)
            actual_effects = composition.get("effects", [])
            
            # Check intensity
            intensity_match = abs(actual_intensity - test_case["expected_intensity"]) < 0.2
            
            # Check effects
            effects_match = all(effect in actual_effects for effect in test_case["expected_effects"])
            
            return {
                "name": test_case["name"],
                "query": test_case["query"],
                "expected_intensity": test_case["expected_intensity"],
                "actual_intensity": actual_intensity,
                "expected_effects": test_case["expected_effects"],
                "actual_effects": actual_effects,
                "intensity_match": intensity_match,
                "effects_match": effects_match,
                "output_path": output_path,
                "composition_path": composition_path,
                "success": intensity_match and effects_match
            }
            
        except Exception as e:
            return {
                "name": test_case["name"],
                "query": test_case["query"],
                "error": str(e),
                "success": False
            }
            
    def _generate_report(self, results: list):
        """Generate a test report.
        
        Args:
            results: List of test results
        """
        report_path = os.path.join(self.output_dir, "test_report.json")
        
        # Calculate statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get("success", False))
        failed_tests = total_tests - successful_tests
        
        # Generate report
        report = {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0
            },
            "results": results
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        print("\nTest Report Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {report['summary']['success_rate']:.2%}")
        
        # Print detailed results
        print("\nDetailed Results:")
        for result in results:
            print(f"\nTest: {result['name']}")
            print(f"Query: {result['query']}")
            if result.get("success", False):
                print("Status: PASSED")
                print(f"Expected Intensity: {result['expected_intensity']}")
                print(f"Actual Intensity: {result['actual_intensity']}")
                print(f"Expected Effects: {result['expected_effects']}")
                print(f"Actual Effects: {result['actual_effects']}")
            else:
                print("Status: FAILED")
                if "error" in result:
                    print(f"Error: {result['error']}")
                    
def main():
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize tester
        tester = VideoGenerationTester(output_dir=temp_dir)
        
        # Run tests
        tester.run_all_tests()
        
if __name__ == "__main__":
    main() 