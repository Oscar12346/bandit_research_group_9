# DSAIT4220 - Research In Intelligent Decision Making
### Group 9

This project uses Python 3.9 or higher.

## Setup Instructions

1. Create a virtual environment:
   ```bash
   # Navigate to the project directory
   cd bandit_practice

   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   # Install all dependencies
   pip install -r requirements.txt
   ```

3. Verify the installation:
   ```bash
   # Check if Python packages are installed correctly
   python -c "import matplotlib; import minigrid; import moviepy"
   ```

## Development Tools

The project includes several development tools for code quality:
- `flake8`: For code linting
- `mypy`: For static type checking
- `black`: For code formatting
- `isort`: For import sorting
- `ipython` and `notebook`: For interactive development

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```