SHELL := /bin/bash
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
SCRIPT := main.py
SEED := 1

# Paths for the latest model files
LATEST_DQL_MODEL := $(shell find ./out/DQL -type f -name 'local_qnetwork.pth' | xargs ls -t | head -n 1)
LATEST_PPO_ACTOR_MODEL := $(shell find ./out/PPO -type f -name 'actor_model.pth' | xargs ls -t | head -n 1)
LATEST_PPO_CRITIC_MODEL := $(shell find ./out/PPO -type f -name 'critic_model.pth' | xargs ls -t | head -n 1)

.PHONY: all clean install run dql dql-test ppo ppo-test

all: clean install dql

install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	$(VENV_NAME)/bin/pip install -r requirements.txt

dql:
	@echo "Running DQL training script..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode train --algorithm DQL

dql-test:
	@echo "Running DQL test script with the latest model..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode test --algorithm DQL --local_qnetwork $(LATEST_DQL_MODEL)

ppo:
	@echo "Running PPO training script..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode train --algorithm PPO

ppo-test:
	@echo "Running PPO test script with the latest actor and critic models..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode test --algorithm PPO --actor_model $(LATEST_PPO_ACTOR_MODEL) --critic_model $(LATEST_PPO_CRITIC_MODEL)

clean:
	@echo "Cleaning up Python cache files and virtual environment..."
	rm -rf $(VENV_NAME)
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf out logs generated_configs
