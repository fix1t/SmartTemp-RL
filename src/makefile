# Makefile

# Variables
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
DQL := dqn_trainer.py
PPO := ppo_trainer.py
all: clean install run

install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Activating virtual environment..."
	bash -c "source venv/bin/activate && pip install -r requirements.txt"

run: install dql

dql:
	@echo "Running DQL script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(DQL)"

ppo:
	@echo "Running PPO script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(PPO)"

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

.PHONY: all clean build run



