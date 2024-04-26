SHELL := /bin/bash
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
SCRIPT := main.py
SEED := 42

# Path for the run configurations script
CONFIG_SCRIPT := run_configurations.py
CONFIG_SCRIPT_OUTPUT := out/generated_configs/results
CONFIG_SCRIPT_FOLDER_DQL := out/generated_configs/dql
CONFIG_SCRIPT_FOLDER_PPO := out/generated_configs/ppo
CONFIG_SCRIPT_FOLDER_NN := out/generated_configs/nn
CONFIG_SCRIPT_TIMESTEPS := $(shell echo $$(($$((4 * 24 * 365 * 15)))))
TRAINING_TIMESTEPS := $(shell echo $$(($$((4 * 24 * 365 * 15)))))
TESTING_TIMESTEPS := $(shell echo $$(($$((4 * 24 * 7)))))

# Paths for the latest model files
LATEST_DQL_MODEL := $(shell find ./out/dql -type f -name 'local_qnetwork.pth' | xargs ls -t | head -n 1)
LATEST_PPO_ACTOR_MODEL := $(shell find ./out/ppo -type f -name 'actor_model.pth' | xargs ls -t | head -n 1)
LATEST_PPO_CRITIC_MODEL := $(shell find ./out/ppo -type f -name 'critic_model.pth' | xargs ls -t | head -n 1)

.PHONY: all clean install run dql dql-test ppo ppo-test

all: clean install dql

install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	$(VENV_NAME)/bin/pip install -r requirements.txt

dql:
	@echo "Running DQL training script..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode train --algorithm DQL --total_timesteps $(TRAINING_TIMESTEPS)

dql-test:
	@echo "Running DQL test script with the latest model..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode test --algorithm DQL --local_qnetwork $(LATEST_DQL_MODEL) --total_timesteps $(TESTING_TIMESTEPS)

ppo:
	@echo "Running PPO training script..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode train --algorithm PPO --total_timesteps $(TRAINING_TIMESTEPS)

ppo-test:
	@echo "Running PPO test script with the latest actor and critic models..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode test --algorithm PPO --actor_model $(LATEST_PPO_ACTOR_MODEL) --critic_model $(LATEST_PPO_CRITIC_MODEL) --total_timesteps $(TESTING_TIMESTEPS)

run-configurations-dql:
	@echo "Running dql configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) -hp --agent DQL --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --folder $(CONFIG_SCRIPT_FOLDER_DQL) --output ${CONFIG_SCRIPT_OUTPUT}

run-configurations-ppo:
	@echo "Running ppo configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) -hp --agent PPO --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --folder $(CONFIG_SCRIPT_FOLDER_PPO) --output ${CONFIG_SCRIPT_OUTPUT}

run-configurations-nn-dql:
	@echo "Running ppo configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) -nn --agent DQL --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --folder $(CONFIG_SCRIPT_FOLDER_NN) --output ${CONFIG_SCRIPT_OUTPUT}/nn
run-configurations-nn-ppo:
	@echo "Running ppo configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) -nn --agent PPO --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --folder $(CONFIG_SCRIPT_FOLDER_NN) --output ${CONFIG_SCRIPT_OUTPUT}/nn

clean:
	@echo "Cleaning up Python cache files and virtual environment..."
	rm -rf $(VENV_NAME)
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf out logs generated_configs
