#!/usr/bin/env python3
"""
ProcureLens Application Entry Point

Usage:
    python run.py                   Start the FastAPI backend server
    python run.py --generate-data   Generate sample procurement data
    python run.py --backend-only    Start only the FastAPI backend
    python run.py --frontend-only   Print instructions to start Streamlit frontend
    python run.py --port 8000       Specify the backend port (default: 8000)
    python run.py --help            Show this help message
"""

import argparse
import os
import subprocess
import sys


def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print(f"[INFO] Loaded environment from {env_path}")
        except ImportError:
            print("[WARN] python-dotenv not installed. Skipping .env file loading.")
            print("       Install it with: pip install python-dotenv")
    else:
        print("[INFO] No .env file found. Using default configuration.")
        print("       Copy .env.example to .env to customize settings.")


def generate_data():
    """Run the data generation script."""
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scripts", "generate_data.py"
    )
    if not os.path.exists(script_path):
        print(f"[ERROR] Data generation script not found at: {script_path}")
        print("        Please ensure scripts/generate_data.py exists.")
        sys.exit(1)

    print("[INFO] Generating sample procurement data...")
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode == 0:
        print("[INFO] Data generation completed successfully.")
    else:
        print("[ERROR] Data generation failed.")
        sys.exit(1)


def check_database():
    """Check if the database exists, prompt user to generate data if not."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "procurelens.db")
    if not os.path.exists(db_path):
        print("=" * 60)
        print("[WARN] Database not found at: data/procurelens.db")
        print()
        print("  To generate sample data, run:")
        print("    python run.py --generate-data")
        print()
        print("  The application will start, but some features may not")
        print("  work until the database is created.")
        print("=" * 60)
        print()


def start_backend(host: str, port: int):
    """Start the FastAPI backend server using uvicorn."""
    print(f"[INFO] Starting ProcureLens FastAPI backend on {host}:{port}")
    print()

    try:
        import uvicorn
        uvicorn.run(
            "app.backend.api:app",
            host=host,
            port=port,
            reload=True,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )
    except ImportError:
        print("[ERROR] uvicorn is not installed.")
        print("        Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to start backend: {e}")
        sys.exit(1)


def print_frontend_instructions():
    """Print instructions for starting the Streamlit frontend."""
    streamlit_port = os.getenv("STREAMLIT_PORT", "8501")
    print()
    print("=" * 60)
    print("  ProcureLens Frontend (Streamlit)")
    print("=" * 60)
    print()
    print("  To start the Streamlit dashboard, open a new terminal")
    print("  and run:")
    print()
    print(f"    streamlit run app/frontend/streamlit_app.py --server.port {streamlit_port}")
    print()
    print(f"  The dashboard will be available at: http://localhost:{streamlit_port}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="ProcureLens - Intelligent Procurement Analytics Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                   Start the backend server
  python run.py --generate-data   Generate sample data first
  python run.py --port 9000       Start backend on port 9000
  python run.py --frontend-only   Show Streamlit startup instructions
        """,
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate sample procurement data before starting",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only the FastAPI backend server",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Print instructions to start the Streamlit frontend",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the backend server (default: 8000)",
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  ProcureLens - Intelligent Procurement Analytics Platform")
    print("=" * 60)
    print()

    # Load environment variables
    load_env()

    # Handle --generate-data
    if args.generate_data:
        generate_data()
        if args.frontend_only:
            print_frontend_instructions()
            return
        if not args.backend_only and not args.frontend_only:
            # Continue to start backend after generating data
            pass

    # Handle --frontend-only
    if args.frontend_only:
        print_frontend_instructions()
        return

    # Check database existence
    check_database()

    # Determine host and port
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = args.port or int(os.getenv("APP_PORT", "8000"))

    # Print frontend instructions before starting backend
    if not args.backend_only:
        print_frontend_instructions()

    # Start the backend
    start_backend(host, port)


if __name__ == "__main__":
    main()
