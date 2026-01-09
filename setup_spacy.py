#!/usr/bin/env python3
"""
spaCy Model Setup Script for Unfold Knowledge Graph

Run this script when network access is available to download
the spaCy English model for improved dependency parsing.

Usage:
    python setup_spacy.py

This will:
1. Check if spaCy is installed
2. Download the en_core_web_sm model
3. Verify the installation
4. Run a quick test to confirm it works
"""

import subprocess
import sys


def check_spacy_installed():
    """Check if spaCy is installed."""
    try:
        import spacy
        print(f"✓ spaCy is installed (version {spacy.__version__})")
        return True
    except ImportError:
        print("✗ spaCy is not installed")
        return False


def check_model_installed():
    """Check if the English model is already installed."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print(f"✓ en_core_web_sm model is already installed")
        return True
    except OSError:
        print("○ en_core_web_sm model is not installed")
        return False


def install_spacy():
    """Install spaCy via pip."""
    print("\nInstalling spaCy...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "spacy"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ spaCy installed successfully")
        return True
    else:
        print(f"✗ Failed to install spaCy: {result.stderr}")
        return False


def download_model():
    """Download the English spaCy model."""
    print("\nDownloading en_core_web_sm model...")
    print("(This may take a few minutes depending on your connection)")

    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Model downloaded successfully")
        return True
    else:
        print(f"✗ Failed to download model")
        print(f"  Error: {result.stderr[:500]}")
        print("\n  If behind a proxy, try:")
        print("    export HTTP_PROXY=http://your-proxy:port")
        print("    export HTTPS_PROXY=http://your-proxy:port")
        return False


def test_model():
    """Run a quick test to verify the model works."""
    print("\nTesting model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")

        # Test sentence
        doc = nlp("Vaswani et al. introduced the Transformer architecture at Google.")

        print("  Test sentence: 'Vaswani et al. introduced the Transformer...'")
        print(f"  Tokens: {[t.text for t in doc][:8]}...")
        print(f"  POS tags: {[t.pos_ for t in doc][:8]}...")

        # Show dependencies
        print("  Dependencies:")
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj', 'pobj'):
                print(f"    {token.text} --[{token.dep_}]--> {token.head.text}")

        # Show entities
        if doc.ents:
            print(f"  Named entities: {[(e.text, e.label_) for e in doc.ents]}")

        print("\n✓ Model is working correctly!")
        return True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def show_usage():
    """Show how to use spaCy in the Unfold modules."""
    print("\n" + "=" * 60)
    print(" USAGE IN UNFOLD")
    print("=" * 60)
    print("""
With the spaCy model installed, the following modules will
automatically use it for improved extraction:

1. Dependency Parsing (backend/app/services/graph/dependency_parsing.py)
   - Uses spaCy's dependency parser for accurate SVO extraction
   - Falls back to pattern-based parsing if unavailable

2. Coreference Resolution (backend/app/services/graph/coreference.py)
   - Can use spaCy for better entity detection
   - Improved pronoun resolution with syntactic context

3. Integrated Pipeline (test_integrated.py)
   - All modules automatically detect and use spaCy

To verify spaCy is being used, look for:
   [DependencyParser] Using spaCy for parsing

If you see:
   [DependencyParser] spaCy unavailable, using pattern-based parsing

Then the model is not installed or there's an import error.
""")


def main():
    print("=" * 60)
    print(" spaCy Model Setup for Unfold Knowledge Graph")
    print("=" * 60)
    print()

    # Step 1: Check spaCy
    if not check_spacy_installed():
        if not install_spacy():
            print("\nSetup failed. Please install spaCy manually:")
            print("  pip install spacy")
            return 1

    # Step 2: Check model
    if check_model_installed():
        # Already installed, just test it
        test_model()
        show_usage()
        return 0

    # Step 3: Download model
    if not download_model():
        print("\nSetup incomplete. Model download failed.")
        print("Try running manually:")
        print("  python -m spacy download en_core_web_sm")
        return 1

    # Step 4: Test model
    if not test_model():
        print("\nSetup incomplete. Model test failed.")
        return 1

    show_usage()
    print("\n✓ Setup complete! spaCy is ready for use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
