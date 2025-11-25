<p align="center">
  <img src="./assets/images/rl4nut-project.png" alt="RL4NUT - Practical Policy Gradient Implementation Starter Kit" width="650"/>
</p>

## Overview

RL4nuts is a collection of reinforcement learning artifacts and scripts for common RL algorithms.

## Setup Instructions

This project uses [Poetry](https://python-poetry.org/) for dependency and environment management.  
It is recommended to use [pyenv](https://github.com/pyenv/pyenv) to manage your Python versions.

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/rl4nuts.git
   cd rl4nuts
   ```

2. **(Optional) Set Python version with pyenv:**
   ```sh
   pyenv install 3.12.10
   pyenv local 3.12.10
   ```

3. **Install dependencies with Poetry:**
   ```sh
   poetry install
   ```

4. **Activate the Poetry environment:**
   ```sh
   poetry shell
   ```

## Usage

To run a script (e.g., `reinforce.py`):
```sh
python rl4nuts/reinforce.py
```
Or, without activating the shell:
```sh
poetry run python rl4nuts/reinforce.py
```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
