#!/usr/bin/env python3
"""
Test script to verify the video-sr conda environment setup
"""

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas version: {pd.__version__}")
        
        import scipy
        print(f"✓ SciPy version: {scipy.__version__}")
        
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        
        import pysindy
        print(f"✓ PySINDy version: {pysindy.__version__}")
        
        import yaml
        print(f"✓ PyYAML version: {yaml.__version__}")
        
        print("\n🎉 All packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_opencv_functionality():
    """Test basic OpenCV functionality."""
    try:
        # Test basic OpenCV operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("✓ OpenCV basic operations working")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_pysindy_functionality():
    """Test basic PySINDy functionality."""
    try:
        # Test basic PySINDy operations
        library = pysindy.PolynomialLibrary(degree=2)
        print("✓ PySINDy basic operations working")
        return True
    except Exception as e:
        print(f"❌ PySINDy test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing video-sr conda environment...\n")
    
    # Import all packages at the top level so they're available in all functions
    try:
        import cv2
        import numpy as np
        import pysindy
        print("✓ All packages imported at top level")
    except ImportError as e:
        print(f"❌ Failed to import packages at top level: {e}")
        exit(1)
    
    imports_ok = test_imports()
    opencv_ok = test_opencv_functionality()
    pysindy_ok = test_pysindy_functionality()
    
    print(f"\n{'='*50}")
    if all([imports_ok, opencv_ok, pysindy_ok]):
        print("🎯 Environment test PASSED - Ready to run video physics analysis!")
    else:
        print("⚠️  Environment test FAILED - Check your conda environment setup")
    print(f"{'='*50}")
