# Contributing to ProcureLens

Thank you for your interest in contributing to ProcureLens! This document provides guidelines and instructions for contributing to this project.

## How to Contribute

1. **Report Bugs**: Open an issue describing the bug, including steps to reproduce, expected behavior, and actual behavior.
2. **Suggest Features**: Open an issue with the `enhancement` label describing the feature and its use case.
3. **Submit Code**: Fork the repository, create a feature branch, and submit a pull request.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ProcureLens
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Generate sample data**:
   ```bash
   python run.py --generate-data
   ```

6. **Run the application**:
   ```bash
   # Start the backend
   python run.py

   # In a separate terminal, start the frontend
   streamlit run app/frontend/streamlit_app.py
   ```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style.
- Use type hints for function signatures.
- Write docstrings for all public modules, classes, and functions.
- Keep functions focused and concise — aim for single responsibility.
- Use meaningful variable and function names.
- Maximum line length is 100 characters.

## Testing

- Write tests for all new features and bug fixes.
- Place tests in the `tests/` directory, mirroring the project structure.
- Use `pytest` as the test runner:
  ```bash
  pytest
  ```
- Run tests with coverage:
  ```bash
  pytest --cov=app tests/
  ```
- Ensure all existing tests pass before submitting a pull request.

## Pull Requests

1. **Fork** the repository and create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** in small, focused commits with clear commit messages.
3. **Write or update tests** to cover your changes.
4. **Run the test suite** and ensure all tests pass.
5. **Update documentation** if your changes affect the public API or user-facing features.
6. **Submit the pull request** with a clear title and description that explains:
   - What the change does
   - Why the change is needed
   - Any relevant issue numbers (e.g., `Closes #42`)
7. **Respond to feedback** during the review process promptly.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for everyone. All participants in this project are expected to:

- **Be respectful**: Treat everyone with respect and consideration. Disagreements are natural, but remain professional.
- **Be inclusive**: Welcome people of all backgrounds and experience levels. Use inclusive language.
- **Be constructive**: Provide constructive feedback. Focus on the idea, not the person.
- **Be collaborative**: Work together to resolve conflicts and build consensus.
- **No harassment**: Harassment, discrimination, or abusive behavior of any kind will not be tolerated.

Violations of this code of conduct may result in removal from the project. If you experience or witness unacceptable behavior, please report it by opening an issue or contacting the maintainers directly.
