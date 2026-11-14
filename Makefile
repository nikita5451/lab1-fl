IMAGE       ?= fl-lab-idf:v5.5.1
PORT        ?= /dev/ttyACM0
DATA        ?= data/data_0.bin
DATA_OFFSET ?= 0x110000
RUN         ?= flwr-virtual
VENV        := .venv
VPY         ?= $(CURDIR)/$(VENV)/bin/python
CORE        := $(VENV)/.core
FULL        := $(VENV)/.full

.PHONY: venv venv-core image build flash flash-data monitor selftest sim-test \
        one-round measure fed-virtual fed-board ref-virtual ref-sim check \
        plot data-stats

$(CORE): host/requirements-core.txt
	python3 -m venv $(VENV)
	$(VPY) -m pip install -r host/requirements-core.txt
	touch $@
$(FULL): $(CORE) host/requirements.txt
	$(VPY) -m pip install -r host/requirements.txt
	touch $@
venv-core: $(CORE)
venv: $(FULL)

image:
	docker build -t $(IMAGE) docker

build:
	docker run --rm --user $(shell id -u):$(shell id -g) -e HOME=/tmp \
	  -v "$(CURDIR)":/project -w /project/firmware $(IMAGE) idf.py build

flash: $(CORE)
	cd firmware/build && $(VPY) -m esptool --chip esp32s3 -p $(PORT) -b 460800 write_flash @flash_args

flash-data: $(CORE)
	$(VPY) -m esptool --chip esp32s3 -p $(PORT) write_flash $(DATA_OFFSET) $(DATA)

monitor: $(CORE)
	$(VPY) -m serial.tools.miniterm $(PORT) 115200

selftest: $(CORE)
	cd firmware/host_test && $(MAKE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.fl_model_ref

sim-test: $(CORE)
	cd firmware/host_test && $(MAKE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.tests.test_protocol_sim

ref-virtual: $(CORE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.run_fedavg --config config/virtual.toml

ref-sim: sim-test
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.run_fedavg --config config/sim.toml

data-stats: $(CORE)
	$(VPY) scripts/data_stats.py

one-round: $(CORE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.one_round --transport serial --port $(PORT)

measure: $(CORE)
	$(VPY) scripts/measure_device.py --port $(PORT)

FLWR = cd host && $(CURDIR)/$(VENV)/bin/flwr run . local-simulation --run-config

fed-virtual: $(FULL)
	$(FLWR) "repo-root='$(CURDIR)' results-dir='results/$(RUN)' board-partition=-1"

fed-board: $(FULL)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.board_stats reset --port $(PORT)
	$(FLWR) "repo-root='$(CURDIR)' results-dir='results/$(RUN)' board-partition=0 board-port='$(PORT)'"
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.board_stats read --port $(PORT) --out ../results/$(RUN)/board_stats.json

check: $(CORE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.check_federation ../results/$(RUN)

plot: $(CORE)
	$(VPY) scripts/plot_rounds.py results/$(RUN)
