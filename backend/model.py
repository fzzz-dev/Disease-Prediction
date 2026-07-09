import pickle
import sys

def check_dependencies():
    """Check what packages are required to load the model"""
    with open('disease_predictor_model.pkl', 'rb') as f:
        # Read the pickle file and extract module information
        import pickletools
        
        def find_imports():
            imports = set()
            with open('disease_predictor_model.pkl', 'rb') as f:
                # Look for module names in the pickle
                for opcode, arg, pos in pickletools.genops(f):
                    if opcode.name == 'SHORT_BINUNICODE' or opcode.name == 'BINUNICODE':
                        if arg.startswith('sklearn') or arg.startswith('numpy') or arg.startswith('joblib'):
                            imports.add(arg)
            return imports
        
        imports = find_imports()
        print("Required packages found in pickle:")
        for imp in imports:
            print(f"  - {imp}")

if __name__ == "__main__":
    check_dependencies()