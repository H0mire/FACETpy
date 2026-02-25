#!/bin/sh
set -eu

REPO_URL="${FACETPY_REPO_URL:-https://github.com/H0mire/facetpy.git}"
TARGET_DIR="${FACETPY_INSTALL_DIR:-facetpy}"
BRANCH="${FACETPY_REPO_BRANCH:-}"

setup_colors() {
  if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    ESC="$(printf '\033')"
    BOLD="${ESC}[1m"
    DIM="${ESC}[2m"
    RED="${ESC}[31m"
    GREEN="${ESC}[32m"
    YELLOW="${ESC}[33m"
    BLUE="${ESC}[34m"
    CYAN="${ESC}[36m"
    RESET="${ESC}[0m"
  else
    BOLD=""
    DIM=""
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    CYAN=""
    RESET=""
  fi
}

banner() {
  printf '%b\n' "${CYAN}${BOLD}"
  cat <<'EOF'
    _________   __________________
   / ____/   | / ____/ ____/_  __/___  __  __
  / /_  / /| |/ /   / __/   / / / __ \/ / / /
 / __/ / ___ / /___/ /___  / / / /_/ / /_/ /
/_/   /_/  |_\____/_____/ /_/ / .___/\__, /
                             /_/    /____/
EOF
  printf '%b\n' "${RESET}${DIM}FACETpy Bootstrap${RESET}"
}

info() { printf '%b\n' "${BLUE}${BOLD}[INFO]${RESET} $*"; }
warn() { printf '%b\n' "${YELLOW}${BOLD}[WARN]${RESET} $*"; }
ok() { printf '%b\n' "${GREEN}${BOLD}[OK]${RESET} $*"; }
fail() { printf '%b\n' "${RED}${BOLD}[ERROR]${RESET} $*" >&2; }

usage() {
  cat <<'EOF'
Usage: bootstrap.sh [bootstrap-options] [-- install-options]

Bootstrap options:
  --dir <path>      Target directory for cloning (default: ./facetpy)
  --repo <url>      Repository URL
  --branch <name>   Git branch or tag to clone
  -h, --help        Show this help message

Everything after '--' is passed to scripts/install.sh.
Example:
  curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh -s -- -- --yes -E all
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command not found: $1"
    exit 1
  fi
}

clone_or_reuse_repo() {
  if [ -d "${TARGET_DIR}/.git" ]; then
    info "Using existing repository at ${TARGET_DIR}"
    return 0
  fi

  if [ -e "${TARGET_DIR}" ]; then
    fail "Target path '${TARGET_DIR}' exists and is not a FACETpy git repository."
    fail "Choose another directory with --dir <path>."
    exit 1
  fi

  info "Cloning FACETpy into ${TARGET_DIR}..."
  if [ -n "${BRANCH}" ]; then
    git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${TARGET_DIR}"
  else
    git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
  fi
  ok "Repository clone completed."
}

main() {
  setup_colors
  banner
  require_cmd git
  require_cmd bash

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --dir)
        if [ "$#" -lt 2 ]; then
          fail "Error: --dir requires a value."
          exit 1
        fi
        TARGET_DIR="$2"
        shift 2
        ;;
      --repo)
        if [ "$#" -lt 2 ]; then
          fail "Error: --repo requires a value."
          exit 1
        fi
        REPO_URL="$2"
        shift 2
        ;;
      --branch)
        if [ "$#" -lt 2 ]; then
          fail "Error: --branch requires a value."
          exit 1
        fi
        BRANCH="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      *)
        # Unknown option is assumed to belong to scripts/install.sh.
        break
        ;;
    esac
  done

  clone_or_reuse_repo
  cd "${TARGET_DIR}"

  if [ ! -f "./scripts/install.sh" ]; then
    fail "./scripts/install.sh not found in cloned repository."
    exit 1
  fi

  info "Running FACETpy installer..."
  bash ./scripts/install.sh "$@"
  ok "Bootstrap finished."
}

main "$@"
