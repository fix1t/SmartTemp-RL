# Makefile

# Variables
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
SCRIPT := main.py
MODEL_PPO := 04-05_13-10
MODEL_DQL := 04-05_13-53

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
	$(PYTHON) $(SCRIPT) --mode train --algorithm DQL"

dql-test:
	@echo "Running DQL test script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(SCRIPT) --mode test --algorithm DQL --local_qnetwork out/DQL/${MODEL_DQL}_local_qnetwork.pth"

ppo:
	@echo "Running PPO script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(SCRIPT) --mode train --algorithm PPO"

ppo-test:
	@echo "Running PPO test script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(SCRIPT) --mode test --algorithm PPO --actor_model out/PPO/${MODEL_PPO}_actor_model.pth --critic_model out/PPO/${MODEL_PPO}_critic_model.pth"

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

.PHONY: all clean build run
