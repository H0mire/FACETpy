#!/usr/bin/env bash
set -euo pipefail

MIN_POETRY_VERSION="1.8.0"
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
  cat <<EOF
${GREEN}${BOLD}
    _________   __________________               _            __        ____         __   __
   / ____/   | / ____/ ____/_  __/___  __  __   (_)___  _____/ /_____ _/ / /__  ____/ /  / /
  / /_  / /| |/ /   / __/   / / / __ \/ / / /  / / __ \/ ___/ __/ __ `/ / / _ \/ __  /  / / 
 / __/ / ___ / /___/ /___  / / / /_/ / /_/ /  / / / / (__  ) /_/ /_/ / / /  __/ /_/ /  /_/  
/_/   /_/  |_\____/_____/ /_/ / .___/\__, /  /_/_/ /_/____/\__/\__,_/_/_/\___/\__,_/  (_)   
                             /_/    /____/                                                      
${RESET}
${CYAN}${BOLD}FACETpy was installed successfully.${RESET}
${DIM}
Next steps:
  1) Run examples:
       poetry run python examples/quickstart.py
       poetry run python examples/evaluation.py
  2) Build optional ANC extension:
       poetry run build-fastranc
  3) Optional extras:
       ./scripts/install.sh -E all
${RESET}
EOF
}

info() { printf '%b\n' "${BLUE}${BOLD}[INFO]${RESET} $*"; }
warn() { printf '%b\n' "${YELLOW}${BOLD}[WARN]${RESET} $*"; }
ok() { printf '%b\n' "${GREEN}${BOLD}[OK]${RESET} $*"; }
fail() { printf '%b\n' "${RED}${BOLD}[ERROR]${RESET} $*" >&2; }

usage() {
  cat <<'EOF'
Usage: ./scripts/install.sh [options]

Options:
  -y, --yes             Non-interactive mode (auto-confirm Poetry installation)
  -E, --extras <name>   Install one optional extra (repeatable)
  -h, --help            Show this help message
EOF
}

version_gte() {
  local current="$1"
  local minimum="$2"
  local IFS='.'
  local i
  local current_parts=()
  local minimum_parts=()
  local length=0

  read -r -a current_parts <<< "$current"
  read -r -a minimum_parts <<< "$minimum"

  if [[ "${#current_parts[@]}" -gt "${#minimum_parts[@]}" ]]; then
    length="${#current_parts[@]}"
  else
    length="${#minimum_parts[@]}"
  fi

  for ((i=0; i<length; i+=1)); do
    local c="${current_parts[i]:-0}"
    local m="${minimum_parts[i]:-0}"

    if ((10#${c} > 10#${m})); then
      return 0
    fi
    if ((10#${c} < 10#${m})); then
      return 1
    fi
  done

  return 0
}

extract_poetry_version() {
  local raw="$1"
  local parsed
  parsed="$(printf '%s\n' "$raw" | sed -E 's/.*version ([0-9]+(\.[0-9]+){1,2}).*/\1/')"

  if [[ ! "$parsed" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]; then
    return 1
  fi

  echo "$parsed"
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
    3.11|3.12)
      ok "Detected Python ${version} (${python_bin})."
      ;;
    *)
      fail "FACETpy requires Python 3.11 or 3.12, found ${version} (${python_bin})."
      fail "Please install a supported Python version and re-run this script."
      exit 1
      ;;
  esac
}

poetry_bin() {
  if command -v poetry >/dev/null 2>&1; then
    command -v poetry
    return 0
  fi

  if [[ -x "${HOME}/.local/bin/poetry" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "${HOME}/.local/bin/poetry"
    return 0
  fi

  return 1
}

install_poetry() {
  local python_bin="$1"
  info "Installing Poetry using the official installer..."

  if command -v curl >/dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org | "$python_bin" -
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://install.python-poetry.org | "$python_bin" -
  else
    fail "Need curl or wget to install Poetry automatically."
    exit 1
  fi
}

ensure_poetry() {
  local python_bin="$1"
  local pbin

  if pbin="$(poetry_bin)"; then
    local pversion
    if ! pversion="$(extract_poetry_version "$("$pbin" --version)")"; then
      fail "Could not parse Poetry version from '$("$pbin" --version)'."
      exit 1
    fi
    if ! version_gte "$pversion" "$MIN_POETRY_VERSION"; then
      fail "Poetry ${pversion} detected, but ${MIN_POETRY_VERSION}+ is required."
      fail "Please update Poetry and run this script again."
      exit 1
    fi
    ok "Detected Poetry ${pversion}."
    return 0
  fi

  local answer="n"
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    answer="y"
  else
    local poetry_prompt
    poetry_prompt="$(cat <<EOF
${YELLOW}${BOLD}
Poetry is required to install FACETpy.
${RESET}${DIM}
Why this is needed:
  - Poetry creates and manages the project virtual environment
  - Poetry installs all pinned dependencies from pyproject.toml/poetry.lock
  - FACETpy commands in this repo are expected to run via 'poetry run ...'

Install Poetry now using the official installer? [y/N]
${RESET}
EOF
)"

    if [[ -t 0 ]]; then
      printf '%b' "$poetry_prompt"
      read -r answer
    elif [[ -r /dev/tty ]]; then
      printf '%b' "$poetry_prompt" >/dev/tty
      read -r answer </dev/tty
    else
      fail "Poetry is not installed, but no interactive terminal is available for confirmation."
      fail "Re-run with --yes to auto-install Poetry, or install Poetry manually."
      exit 1
    fi
  fi

  local answer_lower
  answer_lower="$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')"

  case "${answer_lower}" in
    y|yes)
      install_poetry "$python_bin"
      ;;
    *)
      warn "Aborted: Poetry is required to install FACETpy."
      exit 1
      ;;
  esac

  if ! pbin="$(poetry_bin)"; then
    fail "Poetry installation completed, but command not found in PATH."
    fail "Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
  fi

  local pversion
  if ! pversion="$(extract_poetry_version "$("$pbin" --version)")"; then
    fail "Could not parse Poetry version from '$("$pbin" --version)'."
    exit 1
  fi
  if ! version_gte "$pversion" "$MIN_POETRY_VERSION"; then
    fail "Poetry ${pversion} detected, but ${MIN_POETRY_VERSION}+ is required."
    exit 1
  fi
  ok "Detected Poetry ${pversion}."
}

install_dependencies() {
  local args=(install --no-interaction)
  local extra

  for extra in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do
    args+=(-E "$extra")
  done

  poetry "${args[@]}"
}

main() {
  setup_colors
  banner
  parse_args "$@"

  local python_bin
  if ! python_bin="$(find_python)"; then
    fail "Python was not found."
    fail "Install Python 3.11 or 3.12, then re-run ./scripts/install.sh."
    exit 1
  fi

  info "Checking Python and Poetry prerequisites..."
  check_python_version "$python_bin"
  ensure_poetry "$python_bin"
  info "Installing FACETpy dependencies with Poetry..."
  install_dependencies

  printf '\n'
  ok "FACETpy dependencies installed."
  if [[ "${#EXTRAS[@]}" -gt 0 ]]; then
    ok "Installed extras: ${EXTRAS[*]}"
  else
    info "Tip: install optional extras with ./scripts/install.sh -E all"
  fi
  info "Next step (optional): poetry run build-fastranc"
  printf '\n'
  farewell
}

main "$@"
