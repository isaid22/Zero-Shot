"""
Basic validation tests for zero-shot examples.

These tests verify that the example files have correct structure
and can be imported without errors (without running the heavy models).
"""

import ast
import sys

def test_file_syntax(filename):
    """Test if a Python file has valid syntax."""
    try:
        with open(filename, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"✓ {filename}: Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ {filename}: Syntax error - {e}")
        return False

def test_file_structure(filename, expected_elements):
    """Test if a Python file contains expected functions/classes."""
    try:
        with open(filename, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        
        # Extract function names
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        missing = set(expected_elements) - set(functions)
        if missing:
            print(f"✗ {filename}: Missing expected elements: {missing}")
            return False
        else:
            print(f"✓ {filename}: Contains all expected elements")
            return True
    except Exception as e:
        print(f"✗ {filename}: Error checking structure - {e}")
        return False

def test_has_docstring(filename):
    """Test if a Python file has a module docstring."""
    try:
        with open(filename, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        
        docstring = ast.get_docstring(tree)
        if docstring:
            print(f"✓ {filename}: Has module docstring")
            return True
        else:
            print(f"✗ {filename}: Missing module docstring")
            return False
    except Exception as e:
        print(f"✗ {filename}: Error checking docstring - {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Zero-Shot Examples Validation Tests")
    print("=" * 60)
    print()
    
    all_tests_passed = True
    
    # Test files
    test_files = [
        ("text_classification_example.py", ["main"]),
        ("image_classification_example.py", ["main", "load_image_from_url"]),
        ("question_answering_example.py", ["main"]),
    ]
    
    for filename, expected_functions in test_files:
        print(f"\nTesting {filename}:")
        print("-" * 60)
        
        # Test syntax
        if not test_file_syntax(filename):
            all_tests_passed = False
            continue
        
        # Test structure
        if not test_file_structure(filename, expected_functions):
            all_tests_passed = False
        
        # Test docstring
        if not test_has_docstring(filename):
            all_tests_passed = False
    
    # Test requirements.txt exists and is readable
    print("\nTesting requirements.txt:")
    print("-" * 60)
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.readlines()
        if requirements:
            print(f"✓ requirements.txt: Found {len(requirements)} dependencies")
        else:
            print("✗ requirements.txt: File is empty")
            all_tests_passed = False
    except Exception as e:
        print(f"✗ requirements.txt: Error reading file - {e}")
        all_tests_passed = False
    
    # Summary
    print()
    print("=" * 60)
    if all_tests_passed:
        print("✓ All validation tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
