#!/usr/bin/env bash
set -e

SKILL_NAME="talking-head-autocut"
TARGET_DIR="${HOME}/.gemini/config/skills/${SKILL_NAME}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing ${SKILL_NAME} to ${TARGET_DIR}..."

# Create parent directory if needed
mkdir -p "${HOME}/.gemini/config/skills"

# Check if target already exists
if [ -L "${TARGET_DIR}" ]; then
    echo "Updating existing symlink: ${TARGET_DIR} -> ${REPO_DIR}"
    rm -f "${TARGET_DIR}"
elif [ -d "${TARGET_DIR}" ]; then
    echo "Backing up existing directory: ${TARGET_DIR} -> ${TARGET_DIR}.bak"
    mv "${TARGET_DIR}" "${TARGET_DIR}.bak"
fi

ln -s "${REPO_DIR}" "${TARGET_DIR}"
echo "==> Successfully linked ${REPO_DIR} to ${TARGET_DIR}"
echo "==> The skill is now active in Antigravity!"
