#!/usr/bin/env python3
"""
Unfold Portable - Standalone executable entry point for USB deployment.

This script provides a self-contained version of Unfold that can run from
any location including USB drives without external database dependencies.
"""

import os
import sys
import json
import sqlite3
import webbrowser
import threading
import time
from pathlib import Path
from typing import Optional

# Determine the base directory (where the exe is located)
if getattr(sys, 'frozen', False):
    # Running as PyInstaller executable
    BASE_DIR = Path(sys.executable).parent
else:
    # Running as script
    BASE_DIR = Path(__file__).parent

# Portable data directory
DATA_DIR = BASE_DIR / "unfold_data"
CONFIG_FILE = DATA_DIR / "config.json"
DB_FILE = DATA_DIR / "unfold.db"
GRAPHS_DIR = DATA_DIR / "graphs"
CACHE_DIR = DATA_DIR / "cache"

# Default configuration
DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8080,
    "open_browser": True,
    "debug": False,
    "ollama_host": "http://localhost:11434",
    "llm_provider": "ollama",
    "llm_model": "llama3.2",
}


def ensure_directories():
    """Create necessary directories for portable operation."""
    DATA_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    print(f"✓ Data directory: {DATA_DIR}")


def load_config() -> dict:
    """Load configuration from file or create default."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"⚠ Could not load config: {e}, using defaults")
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"⚠ Could not save config: {e}")


def init_sqlite_db():
    """Initialize SQLite database for portable storage."""
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    # Create tables for documents and knowledge graphs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            text TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            metadata TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            source_text TEXT NOT NULL,
            target_text TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            extraction_method TEXT,
            metadata TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS graphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            name TEXT NOT NULL,
            graph_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✓ Database initialized: {DB_FILE}")


def open_browser_delayed(url: str, delay: float = 1.5):
    """Open browser after a delay to allow server to start."""
    def _open():
        time.sleep(delay)
        print(f"⏳ Opening browser at {url}")
        webbrowser.open(url)

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


def print_banner():
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗   ██╗███╗   ██╗███████╗ ██████╗ ██╗     ██████╗        ║
║   ██║   ██║████╗  ██║██╔════╝██╔═══██╗██║     ██╔══██╗       ║
║   ██║   ██║██╔██╗ ██║█████╗  ██║   ██║██║     ██║  ██║       ║
║   ██║   ██║██║╚██╗██║██╔══╝  ██║   ██║██║     ██║  ██║       ║
║   ╚██████╔╝██║ ╚████║██║     ╚██████╔╝███████╗██████╔╝       ║
║    ╚═════╝ ╚═╝  ╚═══╝╚═╝      ╚═════╝ ╚══════╝╚═════╝        ║
║                                                               ║
║   PDF Knowledge Graph Extraction - Portable Edition           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_ollama():
    """Check if Ollama is available for LLM extraction."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Ollama available ({len(models)} models)")
            return True
    except Exception:
        pass
    print("⚠ Ollama not available (LLM extraction will be disabled)")
    return False


def run_server(config: dict):
    """Run the Unfold server."""
    import uvicorn

    # Set environment variables for portable operation
    os.environ["UNFOLD_PORTABLE"] = "1"
    os.environ["UNFOLD_DATA_DIR"] = str(DATA_DIR)
    os.environ["UNFOLD_DB_FILE"] = str(DB_FILE)
    os.environ["UNFOLD_GRAPHS_DIR"] = str(GRAPHS_DIR)
    os.environ["UNFOLD_CACHE_DIR"] = str(CACHE_DIR)
    os.environ["UNFOLD_LLM_PROVIDER"] = config.get("llm_provider", "ollama")
    os.environ["UNFOLD_LLM_MODEL"] = config.get("llm_model", "llama3.2")

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8080)

    # Open browser if configured
    if config.get("open_browser", True):
        open_browser_delayed(f"http://{host}:{port}")

    print(f"\n✓ Starting server at http://{host}:{port}")
    print("  Press Ctrl+C to stop\n")

    # Import and run the app
    try:
        # Add backend to path
        backend_path = BASE_DIR / "backend"
        if backend_path.exists():
            sys.path.insert(0, str(backend_path))

        from app.main import app
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info" if config.get("debug") else "warning",
        )
    except ImportError as e:
        print(f"✗ Could not import application: {e}")
        print("  Running in standalone extraction mode...")
        run_standalone_extractor(config)


def run_standalone_extractor(config: dict):
    """Run standalone knowledge graph extractor without full server."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import json

    class UnfoldHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(get_standalone_html().encode())
            elif self.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                status = {
                    "status": "running",
                    "version": "portable",
                    "ollama_available": check_ollama(),
                    "data_dir": str(DATA_DIR),
                }
                self.wfile.write(json.dumps(status).encode())
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == "/api/extract":
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data)
                    result = extract_knowledge_graph(data.get("text", ""))
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8080)

    server = HTTPServer((host, port), UnfoldHandler)
    print(f"\n✓ Standalone extractor running at http://{host}:{port}")
    print("  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")


def extract_knowledge_graph(text: str) -> dict:
    """Extract knowledge graph from text using the full backend pipeline."""
    # Add backend to path
    backend_path = str(BASE_DIR / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # Check for required dependencies and provide helpful error messages
    missing_deps = []
    try:
        import pydantic_settings
    except ImportError:
        missing_deps.append("pydantic-settings")

    try:
        import spacy
    except ImportError:
        missing_deps.append("spacy")

    try:
        import sqlalchemy
    except ImportError:
        missing_deps.append("sqlalchemy")

    if missing_deps:
        error_msg = f"Missing required dependencies: {', '.join(missing_deps)}\n"
        error_msg += "\nTo install all dependencies, run:\n"
        error_msg += "  pip install -r requirements_portable.txt\n"
        error_msg += "\nOr install individually:\n"
        for dep in missing_deps:
            error_msg += f"  pip install {dep}\n"

        print(f"\n{'='*60}")
        print("DEPENDENCY ERROR")
        print('='*60)
        print(error_msg)
        print('='*60 + "\n")

        # Fall back to simple Ollama extraction if available
        print("  [Fallback] Attempting Ollama-based extraction...")
        return extract_with_ollama_fallback(text)

    try:
        from app.services.graph.integrated_pipeline import IntegratedRelationExtractor
        from app.services.graph.extractor import extract_entities

        # Extract entities
        print("  [Extraction] Extracting entities...")
        entities = extract_entities(text)
        print(f"  [Extraction] Found {len(entities)} entities")

        # Extract relations
        print("  [Extraction] Extracting relations...")
        extractor = IntegratedRelationExtractor(
            use_coreference=True,
            use_dependency=True,
            use_llm=True,
            use_patterns=True,
        )
        relations = extractor.extract_relations(text, entities)
        print(f"  [Extraction] Found {len(relations)} relations")

        return {
            "success": True,
            "entities": [
                {"text": e.text, "type": e.entity_type.value}
                for e in entities
            ],
            "relations": [
                {
                    "source": r.source_text,
                    "target": r.target_text,
                    "type": r.relation_type.value if hasattr(r.relation_type, 'value') else str(r.relation_type),
                    "confidence": r.confidence,
                    "method": r.extraction_method,
                }
                for r in relations
            ],
        }
    except ImportError as e:
        module_name = str(e).replace("No module named ", "").strip("'\"")
        error_msg = f"Missing module: {module_name}\n"
        error_msg += "\nTo install all dependencies, run:\n"
        error_msg += "  pip install -r requirements_portable.txt"

        print(f"\n{'='*60}")
        print("IMPORT ERROR")
        print('='*60)
        print(error_msg)
        print('='*60 + "\n")

        # Fall back to Ollama
        print("  [Fallback] Attempting Ollama-based extraction...")
        return extract_with_ollama_fallback(text)
    except Exception as e:
        import traceback
        print(f"  [Extraction] Error: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "entities": [],
            "relations": [],
        }


def extract_with_ollama_fallback(text: str) -> dict:
    """Fallback extraction using Ollama when dependencies are missing."""
    try:
        import httpx
    except ImportError:
        return {
            "success": False,
            "error": "Cannot perform extraction: missing dependencies (pydantic-settings, spacy, etc.) and httpx for Ollama fallback.\n\nRun: pip install -r requirements_portable.txt",
            "entities": [],
            "relations": [],
        }

    try:
        config = load_config()
        ollama_url = config.get("ollama_host", "http://localhost:11434")
        model = config.get("llm_model", "llama3.2")

        # Check if Ollama is available
        try:
            check_response = httpx.get(f"{ollama_url}/api/tags", timeout=2.0)
            if check_response.status_code != 200:
                raise Exception("Ollama not responding")
        except Exception:
            return {
                "success": False,
                "error": f"Backend dependencies missing and Ollama not available at {ollama_url}.\n\nEither:\n1. Install dependencies: pip install -r requirements_portable.txt\n2. Or start Ollama: ollama serve",
                "entities": [],
                "relations": [],
            }

        prompt = f"""Extract entities and relationships from the following text.
Return a JSON object with this exact structure:
{{
  "entities": [
    {{"text": "entity name", "type": "PERSON|ORGANIZATION|CONCEPT|METHOD|DATASET|LOCATION|OTHER"}}
  ],
  "relations": [
    {{"source": "entity1", "target": "entity2", "type": "RELATED_TO|USES|CREATES|PART_OF|AFFILIATED_WITH"}}
  ]
}}

Text to analyze:
{text[:3000]}

Return ONLY valid JSON, no other text."""

        print(f"  [Ollama] Using model: {model}")
        response = httpx.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            llm_response = result.get("response", "{}")

            try:
                extracted = json.loads(llm_response)
                entities = extracted.get("entities", [])
                relations = extracted.get("relations", [])

                print(f"  [Ollama] Extracted {len(entities)} entities, {len(relations)} relations")
                return {
                    "success": True,
                    "entities": entities,
                    "relations": [
                        {
                            "source": r.get("source", ""),
                            "target": r.get("target", ""),
                            "type": r.get("type", "RELATED_TO"),
                            "confidence": 0.8,
                            "method": "ollama_fallback",
                        }
                        for r in relations
                    ],
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse Ollama response as JSON: {e}",
                    "entities": [],
                    "relations": [],
                }
        else:
            return {
                "success": False,
                "error": f"Ollama request failed with status {response.status_code}",
                "entities": [],
                "relations": [],
            }
    except httpx.TimeoutException:
        return {
            "success": False,
            "error": f"Ollama request timed out. The model may be loading or the text is too long.",
            "entities": [],
            "relations": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Extraction failed: {str(e)}",
            "entities": [],
            "relations": [],
        }


def get_standalone_html() -> str:
    """Return HTML for standalone extractor interface."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unfold - Knowledge Graph Extractor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
               min-height: 100vh; color: #e0e0e0; padding: 2rem; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 2rem; color: #00d4ff; font-size: 2.5rem; }
        .card { background: rgba(255,255,255,0.05); border-radius: 12px;
                padding: 1.5rem; margin-bottom: 1rem; backdrop-filter: blur(10px); }
        textarea { width: 100%; height: 200px; padding: 1rem; border: 1px solid #333;
                   border-radius: 8px; background: rgba(0,0,0,0.3); color: #fff;
                   font-size: 14px; resize: vertical; }
        button { background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
                 color: #000; border: none; padding: 0.8rem 2rem; border-radius: 8px;
                 cursor: pointer; font-size: 1rem; font-weight: bold; margin-top: 1rem; }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,212,255,0.4); }
        button:disabled { background: #555; cursor: not-allowed; transform: none; }
        .results { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
        .result-box { background: rgba(0,0,0,0.3); border-radius: 8px; padding: 1rem; }
        .result-box h3 { color: #00d4ff; margin-bottom: 0.5rem; }
        .entity { display: inline-block; background: #1e3a5f; padding: 0.3rem 0.6rem;
                  border-radius: 4px; margin: 0.2rem; font-size: 0.9rem; }
        .relation { background: rgba(0,212,255,0.1); padding: 0.5rem; border-radius: 4px;
                    margin: 0.3rem 0; font-size: 0.9rem; border-left: 3px solid #00d4ff; }
        .status { text-align: center; color: #888; font-size: 0.9rem; margin-top: 1rem; }
        .loading { display: none; text-align: center; padding: 2rem; }
        .loading.active { display: block; }
        .spinner { border: 3px solid #333; border-top: 3px solid #00d4ff;
                   border-radius: 50%; width: 30px; height: 30px;
                   animation: spin 1s linear infinite; margin: 0 auto 1rem; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Unfold Knowledge Graph</h1>
        <div class="card">
            <textarea id="textInput" placeholder="Paste your text here to extract knowledge graph..."></textarea>
            <button onclick="extractGraph()" id="extractBtn">Extract Knowledge Graph</button>
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Extracting entities and relations...</p>
        </div>
        <div class="results" id="results" style="display:none;">
            <div class="result-box">
                <h3>Entities (<span id="entityCount">0</span>)</h3>
                <div id="entities"></div>
            </div>
            <div class="result-box">
                <h3>Relations (<span id="relationCount">0</span>)</h3>
                <div id="relations"></div>
            </div>
        </div>
        <div class="status" id="status"></div>
    </div>
    <script>
        async function checkStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('status').textContent =
                    `Status: ${data.status} | Ollama: ${data.ollama_available ? 'Available' : 'Not available'}`;
            } catch (e) {
                document.getElementById('status').textContent = 'Status: Checking...';
            }
        }
        async function extractGraph() {
            const text = document.getElementById('textInput').value.trim();
            if (!text) { alert('Please enter some text'); return; }

            document.getElementById('extractBtn').disabled = true;
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').style.display = 'none';

            try {
                const res = await fetch('/api/extract', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                const data = await res.json();

                if (data.success) {
                    document.getElementById('entityCount').textContent = data.entities.length;
                    document.getElementById('relationCount').textContent = data.relations.length;

                    document.getElementById('entities').innerHTML = data.entities
                        .map(e => `<span class="entity">${e.text} <small>(${e.type})</small></span>`)
                        .join('');

                    document.getElementById('relations').innerHTML = data.relations
                        .map(r => `<div class="relation">${r.source} → <strong>${r.type}</strong> → ${r.target}</div>`)
                        .join('');

                    document.getElementById('results').style.display = 'grid';
                } else {
                    alert('Extraction failed: ' + data.error);
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }

            document.getElementById('extractBtn').disabled = false;
            document.getElementById('loading').classList.remove('active');
        }
        checkStatus();
    </script>
</body>
</html>'''


def main():
    """Main entry point for portable Unfold."""
    print_banner()

    # Setup
    print("Initializing Unfold Portable...")
    print(f"  Base directory: {BASE_DIR}")

    ensure_directories()
    config = load_config()

    # Save config if it doesn't exist
    if not CONFIG_FILE.exists():
        save_config(config)

    init_sqlite_db()
    check_ollama()

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Unfold Portable - Knowledge Graph Extraction")
    parser.add_argument("--host", default=config.get("host"), help="Server host")
    parser.add_argument("--port", type=int, default=config.get("port"), help="Server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Update config with command line args
    config["host"] = args.host
    config["port"] = args.port
    config["open_browser"] = not args.no_browser
    config["debug"] = args.debug

    # Run server
    try:
        run_server(config)
    except KeyboardInterrupt:
        print("\n✓ Unfold stopped")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
