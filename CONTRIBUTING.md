# Contributing to DStretch Python

Thank you for your interest in contributing to DStretch Python! We welcome contributions from everyone. By participating in this project, you agree to abide by our code of conduct (if applicable).

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Key Dependencies

This project relies on several key libraries:
- `numpy`: Numerical operations
- `opencv-python` (cv2): Image processing
- `pytest`: Testing framework
- `ruff`: Linter and formatter
- `pyright`: Static type checker

### Setting up the Development Environment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arqueomendez/dstretch-python.git
    cd dstretch-python
    ```

2.  **Install dependencies using uv:**
    ```bash
    uv sync
    ```
    Or with pip:
    ```bash
    pip install -e .[dev]
    ```

## Development Workflow

### 1. Create a Branch
Create a new branch for your feature or bug fix:
```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes
Write your code! Please ensure you follow existing patterns and keep the codebase clean.

### 3. Code Style and Quality
We use `ruff` for formatting and linting, and `pyright` for type checking.

**Formatting and Linting:**
```bash
uv run ruff check .
uv run ruff format .
```

**Type Checking:**
```bash
uv run pyright
```
Ensure there are no errors before submitting your PR.

### 4. Running Tests
New features should be accompanied by tests. We use `pytest`.

To run all tests:
```bash
uv run pytest
```
If you have added new image processing logic, consider adding an integration test using the images in `test_images/`.

### 5. Commit and Push
We use [Commitizen](https://commitizen-tools.github.io/commitizen/) for commit messages to automate versioning and changelogs.

**Preferred way (`cz` is installed as a dev dependency):**
```bash
uv run cz commit
```
Follow the prompts to create a structured commit message (e.g., `feat: add new colorspace`).

**Manual way (Conventional Commits):**
```bash
git commit -m "feat: add new colorspace to list"
```

### 6. Submit a Pull Request
Push your changes to GitHub and open a Pull Request against the `main` branch. Provide a clear description of your changes and reference any related issues.

## Project Structure

- `src/dstretch`: Main package source code.
    - `colorspaces.py`: Definitions of color space transformations.
    - `decorrelation.py`: Core decorrelation stretch algorithm.
    - `independent_processors.py`: Pre-processing tools (contrast, balance, etc.).
    - `pipeline.py`: Main processing pipeline logic.
    - `gui.py`: Tkinter-based GUI.
- `tests/`: Unit and integration tests.
- `test_images/`: Sample images for testing.

## Reporting Bugs
If you find a bug, please create a GitHub issue with a detailed description, including:
- Steps to reproduce.
- Expected vs. actual behavior.
- Python version and OS.

Thank you for contributing!
