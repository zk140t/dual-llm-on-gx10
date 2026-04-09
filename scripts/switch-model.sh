#!/bin/bash
# Switch the gemma4 container to a different model
# Usage: ./switch-model.sh <script-to-run>
# Example: ./switch-model.sh start-gemma4-moe.sh
#
# Stops and removes current gemma4 container, then runs the specified start script.

SCRIPT="${1:-}"

if [ -z "$SCRIPT" ]; then
  echo "Usage: $0 <start-script>"
  echo "  e.g: $0 start-gemma4-moe.sh"
  echo "  e.g: $0 start-gemma4-31b-dense.sh"
  exit 1
fi

SCRIPT_PATH="$(dirname "$0")/$SCRIPT"

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Script not found: $SCRIPT_PATH"
  exit 1
fi

echo "Stopping gemma4 container..."
docker stop gemma4 2>/dev/null && docker rm gemma4 2>/dev/null
echo "Stopped."

echo "Starting new model with $SCRIPT..."
bash "$SCRIPT_PATH"
