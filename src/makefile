SHELL := /bin/bash
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
SCRIPT := main.py
SEED := 42

# Path for the run configurations script
EVAL_SCRIPT := run_evaluation.py
CONFIG_SCRIPT := run_configurations.py
CONFIG_GENERATE_SCRIPT := tools/config_generator.py
CONFIG_SCRIPT_OUTPUT := out/results
CONFIG_SCRIPT_FOLDER := out/generated_configs
CONFIG_SCRIPT_TIMESTEPS := $(shell echo $$(($$((4 * 24 * 365 * 15)))))
TRAINING_TIMESTEPS := $(shell echo $$(($$((4 * 24 * 365 * 15)))))
TESTING_TIMESTEPS := $(shell echo $$(($$((4 * 24 * 7)))))

# Paths for the latest model files
LATEST_DQL_MODEL := $(shell find ./out/dql -type f -name 'target_qnetwork.pth' | xargs ls -t | head -n 1)
LATEST_PPO_ACTOR_MODEL := $(shell find ./out/ppo -type f -name 'actor_model.pth' | xargs ls -t | head -n 1)
LATEST_PPO_CRITIC_MODEL := $(shell find ./out/ppo -type f -name 'critic_model.pth' | xargs ls -t | head -n 1)

LATEST_DQL_CONFIG:= $(shell find ./out/dql -type f -name 'config.yaml' | xargs ls -t | head -n 1)
LATEST_PPO_CONFIG:= $(shell find ./out/ppo -type f -name 'config.yaml' | xargs ls -t | head -n 1)

TIMESTAMP := $(shell date +'%Y-%m-%d_%H-%M')

.PHONY: all clean install run dql dql-test ppo ppo-test

all: install

install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	$(VENV_NAME)/bin/pip install -r requirements.txt

dql:
	@echo "Running DQL training script..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode train --algorithm DQL --total_timesteps $(TRAINING_TIMESTEPS)

ppo:
	@echo "Running PPO training script..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode train --algorithm PPO --total_timesteps $(TRAINING_TIMESTEPS)

dql-test:
	@echo "Running DQL test script with the latest model..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode test --algorithm DQL --total_timesteps $(TESTING_TIMESTEPS) --config $(LATEST_DQL_CONFIG) --local_qnetwork $(LATEST_DQL_MODEL)

ppo-test:
	@echo "Running PPO test script with the latest actor and critic models..."
	$(PYTHON) $(SCRIPT) --seed $(SEED) --mode test --algorithm PPO --total_timesteps $(TESTING_TIMESTEPS) --config $(LATEST_PPO_CONFIG) --actor_model $(LATEST_PPO_ACTOR_MODEL) --critic_model $(LATEST_PPO_CRITIC_MODEL)

run-configurations-dql: generate-hp-dql
	@echo "Running dql configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) --agent DQL --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --rpc 81  --folder $(CONFIG_SCRIPT_FOLDER)/hp/dql --output ${CONFIG_SCRIPT_OUTPUT}/hp/dql/$(TIMESTAMP)

run-configurations-ppo: generate-hp-ppo
	@echo "Running ppo configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) --agent PPO --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --rpc 81  --folder $(CONFIG_SCRIPT_FOLDER)/hp/ppo --output ${CONFIG_SCRIPT_OUTPUT}/hp/ppo/$(TIMESTAMP)

run-configurations-nn-dql: generate-nn
	@echo "Running ppo configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) --agent DQL --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --rpc 28  --folder $(CONFIG_SCRIPT_FOLDER)/nn --output ${CONFIG_SCRIPT_OUTPUT}/nn/dql/$(TIMESTAMP)

run-configurations-nn-ppo: generate-nn
	@echo "Running ppo configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) --agent PPO --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --rpc 28  --folder $(CONFIG_SCRIPT_FOLDER)/nn --output ${CONFIG_SCRIPT_OUTPUT}/nn/ppo/$(TIMESTAMP)

run-configurations-top-dql: generate-top-dql
	@echo "Running dql configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) --agent DQL --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --rpc 25  --folder $(CONFIG_SCRIPT_FOLDER)/top/dql --output ${CONFIG_SCRIPT_OUTPUT}/top/dql/$(TIMESTAMP)  -s
	
run-configurations-top-ppo: generate-top-ppo
	@echo "Running ppo configurations with $(CONFIG_SCRIPT_TIMESTEPS) timesteps..."
	$(PYTHON) $(CONFIG_SCRIPT) --agent PPO --timesteps $(CONFIG_SCRIPT_TIMESTEPS) --rpc 25 --folder $(CONFIG_SCRIPT_FOLDER)/top/ppo --output ${CONFIG_SCRIPT_OUTPUT}/top/ppo/$(TIMESTAMP) -s

eval-ppo:
	@echo "Evaluating PPO model..."
	$(PYTHON) $(EVAL_SCRIPT) --algorithm PPO --config $(LATEST_PPO_CONFIG) --output ${CONFIG_SCRIPT_OUTPUT}/eval/ppo/$(TIMESTAMP)

eval-dql:
	@echo "Evaluating DQL model..."
	$(PYTHON) $(EVAL_SCRIPT) --algorithm DQL --config $(LATEST_DQL_CONFIG) --output ${CONFIG_SCRIPT_OUTPUT}/eval/dql/$(TIMESTAMP)

generate-nn:
	@echo "Generating NN configurations..."
	rm -rf $(CONFIG_SCRIPT_FOLDER)/nn
	$(PYTHON) $(CONFIG_GENERATE_SCRIPT) --nn --agent DQL --output $(CONFIG_SCRIPT_FOLDER)/nn

generate-hp-dql:
	@echo "Generating DQL configurations..."
	rm -rf $(CONFIG_SCRIPT_FOLDER)/hp/dql
	$(PYTHON) $(CONFIG_GENERATE_SCRIPT) --hp --agent DQL --output $(CONFIG_SCRIPT_FOLDER)/hp/dql

generate-hp-ppo:
	@echo "Generating PPO configurations..."
	rm -rf $(CONFIG_SCRIPT_FOLDER)/hp/ppo
	$(PYTHON) $(CONFIG_GENERATE_SCRIPT) --hp --agent PPO --output $(CONFIG_SCRIPT_FOLDER)/hp/ppo

generate-top-dql:
	@echo "Generating DQL configurations..."
	rm -rf $(CONFIG_SCRIPT_FOLDER)/top/dql
	$(PYTHON) $(CONFIG_GENERATE_SCRIPT) --top --agent DQL --output $(CONFIG_SCRIPT_FOLDER)/top/dql

generate-top-ppo:	
	@echo "Generating PPO configurations..."
	rm -rf $(CONFIG_SCRIPT_FOLDER)/top/ppo

	$(PYTHON) $(CONFIG_GENERATE_SCRIPT) --top --agent PPO --output $(CONFIG_SCRIPT_FOLDER)/top/ppo

clean:
	@echo "Cleaning up Python cache files and virtual environment..."
	rm -rf $(VENV_NAME)
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf out logs generated_configs
