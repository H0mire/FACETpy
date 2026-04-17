#!/usr/bin/env bash
set -euo pipefail

ASSUME_YES=0
EXTRAS=()

setup_colors() {
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    local esc
    esc="$(printf '\033')"
    BOLD="${esc}[1m"
    DIM="${esc}[2m"
    RED="${esc}[31m"
    GREEN="${esc}[32m"
    YELLOW="${esc}[33m"
    BLUE="${esc}[34m"
    CYAN="${esc}[36m"
    RESET="${esc}[0m"
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
  [[ "${FACETPY_BOOTSTRAP:-0}" -eq 1 ]] && return
  cat <<EOF
${CYAN}${BOLD}
    _________   __________________
   / ____/   | / ____/ ____/_  __/___  __  __
  / /_  / /| |/ /   / __/   / / / __ \/ / / /
 / __/ / ___ / /___/ /___  / / / /_/ / /_/ /
/_/   /_/  |_\____/_____/ /_/ / .___/\__, /
                             /_/    /____/
${RESET}${DIM}
FACETpy Installer
${RESET}
EOF
}

farewell() {
  printf '%b\n' "${GREEN}${BOLD}"
  cat <<'EOF'
    _________   __________________               _            __        ____         __   __
   / ____/   | / ____/ ____/_  __/___  __  __   (_)___  _____/ /_____ _/ / /__  ____/ /  / /
  / /_  / /| |/ /   / __/   / / / __ \/ / / /  / / __ \/ ___/ __/ __ `/ / / _ \/ __  /  / / 
 / __/ / ___ / /___/ /___  / / / /_/ / /_/ /  / / / / (__  ) /_/ /_/ / / /  __/ /_/ /  /_/  
/_/   /_/  |_\____/_____/ /_/ / .___/\__, /  /_/_/ /_/____/\__/\__,_/_/_/\___/\__,_/  (_)   
                             /_/    /____/                                                
EOF
  printf '%b\n' "${RESET}${CYAN}${BOLD}FACETpy was installed successfully.${RESET}"
  printf '%b\n' "${DIM}"
  cat <<'EOF'
Next steps:
  1) Run examples:
       uv run python examples/quickstart.py
       uv run python examples/evaluation.py
  2) Build ANC extension (strongly recommended for fast ANC):
       uv run build-fastranc
  3) Tutorial:
       https://facetpy.readthedocs.io/en/latest/getting_started/tutorial.html
       (local source: docs/source/getting_started/tutorial.rst)
  4) Optional extras:
       ./scripts/install.sh -E all
EOF
  printf '%b\n' "${RESET}"
}

info() { printf '%b\n' "${BLUE}${BOLD}[INFO]${RESET} $*"; }
warn() { printf '%b\n' "${YELLOW}${BOLD}[WARN]${RESET} $*"; }
ok() { printf '%b\n' "${GREEN}${BOLD}[OK]${RESET} $*"; }
fail() { printf '%b\n' "${RED}${BOLD}[ERROR]${RESET} $*" >&2; }

usage() {
  cat <<'EOF'
Usage: ./scripts/install.sh [options]

Options:
  -y, --yes             Non-interactive mode (auto-confirm uv installation)
  -E, --extras <name>   Install one optional extra (repeatable)
  -h, --help            Show this help message
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -y|--yes)
        ASSUME_YES=1
        shift
        ;;
      -E|--extras)
        if [[ $# -lt 2 ]]; then
          fail "Error: --extras requires a value."
          exit 1
        fi
        EXTRAS+=("$2")
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        fail "Error: Unknown argument '$1'"
        usage
        exit 1
        ;;
    esac
  done
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi

  return 1
}

check_python_version() {
  local python_bin="$1"
  local version
  version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

  case "$version" in
    3.11|3.12|3.13)
      ok "Detected Python ${version} (${python_bin})."
      ;;
    *)
      fail "FACETpy requires Python 3.11, 3.12, or 3.13, found ${version} (${python_bin})."
      fail "Please install a supported Python version and re-run this script."
      exit 1
      ;;
  esac
}

uv_bin() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi

  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "${HOME}/.local/bin/uv"
    return 0
  fi

  return 1
}

install_uv() {
  info "Installing uv using the official installer..."

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    fail "Need curl or wget to install uv automatically."
    exit 1
  fi
}

ensure_uv() {
  local uv_path

  if uv_path="$(uv_bin)"; then
    ok "Detected uv at ${uv_path}."
    return 0
  fi

  local answer="n"
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    answer="y"
  else
    local uv_prompt
    uv_prompt="$(cat <<EOF
${YELLOW}${BOLD}
uv is required to install FACETpy.
${RESET}${DIM}
Why this is needed:
  - uv creates and manages the project virtual environment
  - uv installs all pinned dependencies from pyproject.toml/uv.lock
  - FACETpy commands in this repo are expected to run via 'uv run ...'

Install uv now using the official installer? [y/N]
${RESET}
EOF
)"

    if [[ -t 0 ]]; then
      printf '%b' "$uv_prompt"
      read -r answer
    elif [[ -r /dev/tty ]]; then
      printf '%b' "$uv_prompt" >/dev/tty
      read -r answer </dev/tty
    else
      fail "uv is not installed, but no interactive terminal is available for confirmation."
      fail "Re-run with --yes to auto-install uv, or install uv manually."
      exit 1
    fi
  fi

  local answer_lower
  answer_lower="$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')"

  case "${answer_lower}" in
    y|yes)
      install_uv
      ;;
    *)
      warn "Aborted: uv is required to install FACETpy."
      exit 1
      ;;
  esac

  if ! uv_path="$(uv_bin)"; then
    fail "uv installation completed, but command not found in PATH."
    fail "Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
  fi
  ok "Detected uv at ${uv_path}."
}

install_dependencies() {
  local args=(sync --locked)
  local extra
  local use_all_extras=0

  for extra in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do
    if [[ "$extra" == "all" ]]; then
      use_all_extras=1
    else
      args+=(--extra "$extra")
    fi
  done

  if [[ "$use_all_extras" -eq 1 ]]; then
    args+=(--all-extras)
  fi

  uv "${args[@]}"
}

main() {
  setup_colors
  banner
  parse_args "$@"

  local python_bin
  if ! python_bin="$(find_python)"; then
    fail "Python was not found."
    fail "Install Python 3.11, 3.12, or 3.13, then re-run ./scripts/install.sh."
    exit 1
  fi

  info "Checking Python and uv prerequisites..."
  check_python_version "$python_bin"
  ensure_uv
  info "Installing FACETpy dependencies with uv..."
  install_dependencies

  printf '\n'
  ok "FACETpy dependencies installed."
  if [[ "${#EXTRAS[@]}" -gt 0 ]]; then
    ok "Installed extras: ${EXTRAS[*]}"
  else
    info "Tip: install optional extras with ./scripts/install.sh -E all"
  fi
  info "Next step (optional): uv run build-fastranc"
  printf '\n'
  farewell
}

main "$@"
