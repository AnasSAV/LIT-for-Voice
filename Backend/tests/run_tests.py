"""
Test Configuration and Runner Script
Master Test Plan Implementation - Comprehensive Test Execution
"""

import pytest
import sys
import os
from pathlib import Path

# Test execution configuration based on Master Test Plan sections

# Test Categories from Test Plan
TEST_CATEGORIES = {
    "data_integrity": {
        "description": "Data and Database Integrity Testing (Section 3.1.1)",
        "files": ["test_data_integrity.py"],
        "priority": "critical",
        "estimated_time": "10 minutes"
    },
    "function_testing": {
        "description": "Function Testing - ML Models and Audio Processing (Section 3.1.2)", 
        "files": ["test_function_testing.py"],
        "priority": "critical",
        "estimated_time": "30 minutes"
    },
    "performance": {
        "description": "Performance Profiling and Load Testing (Sections 3.1.4 & 3.1.5)",
        "files": ["test_performance_load.py"],
        "priority": "important",
        "estimated_time": "15 minutes"
    },
    "security": {
        "description": "Security and Access Control Testing (Section 3.1.6)",
        "files": ["test_security.py"], 
        "priority": "important",
        "estimated_time": "20 minutes"
    }
}

# Performance Thresholds (from Test Plan Section 3.1.4)
PERFORMANCE_BENCHMARKS = {
    "model_inference_max_time": 10.0,      # seconds - for 30-second audio clips
    "audio_upload_max_time": 5.0,          # seconds - for files under 10MB  
    "cache_retrieval_max_time": 0.05,      # seconds - Redis operations
    "ui_response_max_time": 0.1,           # seconds - User interface response
    "concurrent_user_limit": 10,           # Maximum concurrent users
    "max_memory_usage_mb": 2048           # Maximum memory usage in MB
}

# Test execution functions
def run_category_tests(category: str, verbose: bool = True):
    """Run tests for a specific category."""
    if category not in TEST_CATEGORIES:
        print(f"Unknown test category: {category}")
        print(f"Available categories: {list(TEST_CATEGORIES.keys())}")
        return False
    
    config = TEST_CATEGORIES[category]
    print(f"\n{'='*60}")
    print(f"Running {config['description']}")
    print(f"Priority: {config['priority']}")
    print(f"Estimated time: {config['estimated_time']}")
    print(f"{'='*60}")
    
    # Build pytest arguments
    pytest_args = []
    
    if verbose:
        pytest_args.extend(["-v", "-s"])
    
    # Add specific test files
    for test_file in config["files"]:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            pytest_args.append(str(test_path))
        else:
            print(f"Warning: Test file not found: {test_file}")
    
    if not pytest_args or not any(Path(arg).exists() for arg in pytest_args if Path(arg).suffix == '.py'):
        print(f"No test files found for category: {category}")
        return False
    
    # Run tests
    result = pytest.main(pytest_args)
    return result == 0

def run_critical_tests():
    """Run only critical priority tests."""
    print("\n" + "="*80)
    print("RUNNING CRITICAL PRIORITY TESTS")
    print("="*80)
    
    critical_categories = [cat for cat, config in TEST_CATEGORIES.items() 
                          if config["priority"] == "critical"]
    
    results = {}
    for category in critical_categories:
        results[category] = run_category_tests(category)
    
    # Summary
    print(f"\n{'='*60}")
    print("CRITICAL TESTS SUMMARY")
    print(f"{'='*60}")
    
    for category, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{category}: {status}")
    
    return all(results.values())

def run_all_tests():
    """Run all test categories in order of priority."""
    print("\n" + "="*80) 
    print("RUNNING COMPLETE TEST SUITE")
    print("LIT for Voice - Master Test Plan Implementation")
    print("="*80)
    
    # Run critical tests first
    critical_categories = [cat for cat, config in TEST_CATEGORIES.items() 
                          if config["priority"] == "critical"]
    
    important_categories = [cat for cat, config in TEST_CATEGORIES.items()
                           if config["priority"] == "important"]
    
    all_results = {}
    
    # Critical tests
    print("\n🔴 CRITICAL PRIORITY TESTS")
    for category in critical_categories:
        all_results[category] = run_category_tests(category)
        
    # Important tests
    print("\n🟡 IMPORTANT PRIORITY TESTS") 
    for category in important_categories:
        all_results[category] = run_category_tests(category)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    critical_passed = all(all_results[cat] for cat in critical_categories)
    important_passed = all(all_results[cat] for cat in important_categories if cat in all_results)
    
    print(f"Critical Tests: {'PASS' if critical_passed else 'FAIL'}")
    print(f"Important Tests: {'PASS' if important_passed else 'FAIL'}")
    
    for category, passed in all_results.items():
        config = TEST_CATEGORIES[category]
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {category} ({config['priority']}): {status}")
    
    overall_success = critical_passed and important_passed
    print(f"\nOVERALL RESULT: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return overall_success

def run_performance_benchmarks():
    """Run performance tests with detailed benchmarking."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK TESTING")
    print("="*80)
    
    # Run performance tests with benchmarking
    pytest_args = [
        "-v", "-s",
        "--benchmark-only" if "--benchmark-only" in sys.argv else "",
        "test_performance_load.py::TestPerformanceProfiling",
        "-k", "performance"
    ]
    
    # Remove empty args
    pytest_args = [arg for arg in pytest_args if arg]
    
    result = pytest.main(pytest_args)
    
    print(f"\nPerformance Benchmarks: {'PASS' if result == 0 else 'FAIL'}")
    return result == 0

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*80)
    print("GENERATING TEST REPORT")
    print("="*80)
    
    # Run tests with HTML report generation
    report_dir = Path(__file__).parent / "test_reports"
    report_dir.mkdir(exist_ok=True)
    
    pytest_args = [
        "--html=" + str(report_dir / "test_report.html"),
        "--self-contained-html",
        "--tb=short",
        "-v"
    ]
    
    # Add all test files
    for category, config in TEST_CATEGORIES.items():
        for test_file in config["files"]:
            test_path = Path(__file__).parent / test_file
            if test_path.exists():
                pytest_args.append(str(test_path))
    
    result = pytest.main(pytest_args)
    
    print(f"Test report generated: {report_dir / 'test_report.html'}")
    return result == 0

if __name__ == "__main__":
    """
    Test Runner Script
    
    Usage:
        python run_tests.py                    # Run all tests
        python run_tests.py critical          # Run critical tests only
        python run_tests.py data_integrity    # Run specific category
        python run_tests.py performance       # Run performance tests
        python run_tests.py report           # Generate HTML report
    """
    
    if len(sys.argv) < 2:
        # Run all tests by default
        success = run_all_tests()
        sys.exit(0 if success else 1)
    
    command = sys.argv[1].lower()
    
    if command == "critical":
        success = run_critical_tests()
    elif command == "performance":
        success = run_performance_benchmarks()
    elif command == "report":
        success = generate_test_report()
    elif command in TEST_CATEGORIES:
        success = run_category_tests(command)
    elif command == "all":
        success = run_all_tests()
    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands:")
        print("  critical          - Run critical priority tests")
        print("  performance       - Run performance benchmarks")
        print("  report           - Generate HTML test report")
        print("  all              - Run all tests")
        
        for category in TEST_CATEGORIES:
            config = TEST_CATEGORIES[category]
            print(f"  {category:<15} - {config['description']}")
        
        sys.exit(1)
    
    sys.exit(0 if success else 1)