SHELL := /bin/bash
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
SCRIPT := main.py
SEED := 1


.PHONY: all clean install run dql dql-test ppo ppo-test get_latest_dql_model get_latest_ppo_model

all: clean install get_latest_dql_model get_latest_ppo_model dql

install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Activating virtual environment..."
	bash -c "source venv/bin/activate && pip install -r requirements.txt"

dql:
	@echo "Running DQL script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(SCRIPT) --seed ${SEED} --mode train --algorithm DQL"

dql-test: get_latest_dql_model
	@echo "Running DQL test script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(SCRIPT) --seed ${SEED} --mode test --algorithm DQL --local_qnetwork out/DQL/${MODEL_DQL}_local_qnetwork.pth"

ppo:
	@echo "Running PPO script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(SCRIPT) --seed ${SEED} --mode train --algorithm PPO"

ppo-test: get_latest_ppo_model
	@echo "Running PPO test script..."
	bash -c "source $(VENV_NAME)/bin/activate && \
	$(PYTHON) $(SCRIPT) --seed ${SEED} --mode test --algorithm PPO --actor_model out/PPO/${MODEL_PPO}_actor_model.pth --critic_model out/PPO/${MODEL_PPO}_critic_model.pth"

get_latest_dql_model:
	$(eval MODEL_DQL=$(shell find ./out/DQL -type f -name '*.pth' | sed -E 's|.*\/([0-9-]+_[0-9-]+).*|\1|' | sort -r | head -n1))
	@if [ -z "$(MODEL_DQL)" ]; then \
		echo "Warning: No DQL model files found."; \
		return 1; \
	else \
		echo "Running DQL model for datetime $(MODEL_DQL)"; \
	fi

get_latest_ppo_model:
	$(eval MODEL_PPO=$(shell find ./out/PPO -type f -name '*.pth' | sed -E 's|.*\/([0-9-]+_[0-9-]+).*|\1|' | sort -r | head -n1))
	@if [ -z "$(MODEL_PPO)" ]; then \
		echo "Warning: No PPO model files found."; \
		return 1; \
	else \
		echo "Running PPO model for datetime $(MODEL_PPO)"; \
	fi

clean:
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -exec rm -rf {} +
