"""Entry point for running ai-mem as a module.

Usage:
    python -m ai_mem.server --host 0.0.0.0 --port 8000
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="ai-mem server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    from .server import start_server

    print(f"Starting ai-mem server at http://{args.host}:{args.port}")
    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main() or 0)
