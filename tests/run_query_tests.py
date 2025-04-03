"""
Simple script to run tests for the Query.load_embeddings function.
Run with: python tests/run_query_tests.py
"""
import os
import sys
import traceback
from minedd.query import Query


def test_load_embeddings_success():
    """Test successful loading of embeddings from a pickle file."""
    print("Testing load_embeddings_success...")
    
    # Create query instance
    query = Query()
    
    # Load existing test embeddings
    test_path = os.path.join(os.path.dirname(__file__), "Data/test_embeddings.pkl")
    
    # Verify file exists
    if not os.path.exists(test_path):
        print(f"FAIL: Test pickle file not found at {test_path}")
        return False
    
    try:
        # Load embeddings
        result = query.load_embeddings(test_path)
        
        # Verify docs are loaded and method returns self
        if query.docs is None:
            print("FAIL: Docs should be loaded")
            return False
        
        if result is not query:
            print("FAIL: Method should return self for chaining")
            return False
            
        print("PASS: load_embeddings_success")
        return True
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        traceback.print_exc()
        return False


def test_load_embeddings_file_not_found():
    """Test load_embeddings raises FileNotFoundError when file doesn't exist."""
    print("Testing load_embeddings_file_not_found...")
    
    # Create query instance
    query = Query()
    nonexistent_path = "nonexistent_file.pkl"
    
    try:
        # This should raise FileNotFoundError
        query.load_embeddings(nonexistent_path)
        print("FAIL: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        # Verify error message
        if f"File {nonexistent_path} not found" not in str(e):
            print(f"FAIL: Unexpected error message: {e}")
            return False
        print("PASS: load_embeddings_file_not_found")
        return True
    except Exception as e:
        print(f"FAIL: Wrong error type: {type(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    success = all([
        test_load_embeddings_success(),
        test_load_embeddings_file_not_found()
    ])
    
    # Report results
    if success:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)