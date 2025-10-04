import subprocess
import sys


def install_package(package_name: str) -> bool:
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed: {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install: {package_name}")
        return False


def download_spacy_model() -> bool:
    """Download spaCy English model"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
        print("✅ Successfully downloaded: spaCy English model")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to download: spaCy English model")
        return False


def main() -> None:
    """Install all required libraries for financial document chunking"""

    print("Installing Financial Document Chunking Libraries...")
    print("=" * 60)

    # Core libraries for intelligent chunking
    libraries = [
        "pdfplumber",  # PDF structure analysis and table extraction
        "spacy",  # Financial NLP and entity recognition
        "sentence-transformers",  # Semantic content mapping
        "scikit-learn",  # Text similarity and clustering
        "pandas",  # Data manipulation for financial data
        "numpy",  # Numerical operations
        "nltk",  # Natural language processing utilities
    ]

    # Optional but useful libraries
    optional_libraries = [
        "fuzzywuzzy",  # Fuzzy string matching for financial terms
        "python-levenshtein",  # Fast string distance calculations
        "textstat",  # Text complexity analysis
    ]

    successful_installs = 0
    total_libraries = len(libraries)

    # Install core libraries
    print("\nInstalling Core Libraries:")
    print("-" * 30)
    for library in libraries:
        if install_package(library):
            successful_installs += 1

    # Install optional libraries
    print("\nInstalling Optional Libraries:")
    print("-" * 30)
    for library in optional_libraries:
        install_package(library)

    # Download spaCy model
    print("\nDownloading Language Models:")
    print("-" * 30)
    if successful_installs >= total_libraries - 1:  # Allow for one failure
        download_spacy_model()

    # Final status
    print("\n" + "=" * 60)
    print("Installation Summary:")
    print(f"✅ Core libraries installed: {successful_installs}/{total_libraries}")

    if successful_installs == total_libraries:
        print("✅ All installations successful!")
        print("✅ Ready for financial document chunking")
    else:
        print("⚠️  Some installations failed - check error messages above")

    # Verification
    print("\nVerifying Installations:")
    print("-" * 30)

    verification_imports = {
        "pdfplumber": "pdfplumber",
        "spacy": "spacy",
        "sentence_transformers": "sentence_transformers",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "numpy": "numpy",
    }

    for module_name, package_name in verification_imports.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name} - Ready")
        except ImportError:
            print(f"❌ {package_name} - Import failed")


if __name__ == "__main__":
    main()
