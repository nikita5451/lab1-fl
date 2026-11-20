SN_BIN   := PATH="$(CURDIR)/$(VENV)/bin:$$PATH"
SN       := $(SN_BIN) flower-supernode --insecure --superlink 127.0.0.1:8000
RUNCFG    = repo-root='$(CURDIR)' results-dir='results/$(EXP)' num-rounds=$(ROUNDS) min-nodes=$(MIN) round-delay=$(DELAY)
EXP      ?= exp-b
ROUNDS   ?= 10
MIN      ?= 4
DELAY    ?= 0.0
WEIGHTS  ?= results/exp-b/final_weights.bin

.PHONY: superlink node0 node0-sim node1 node2 node3 board-reset flwr-run board-read \
        board-global check-server lab2-table lab2-plot

superlink: $(FULL)
	$(SN_BIN) flower-superlink --insecure

node0: $(FULL)
	$(SN) --port 9094 --node-config 'partition-id=0 backend="serial" device-port="$(PORT)"'

node0-sim: $(FULL)
	$(SN) --port 9094 --node-config 'partition-id=0 backend="sim" data-path="$(CURDIR)/data/data_0.bin"'

node1: $(FULL)
	$(SN) --port 9095 --node-config 'partition-id=1 backend="sim" data-path="$(CURDIR)/data/data_1.bin"'

node2: $(FULL)
	$(SN) --port 9096 --node-config 'partition-id=2 backend="sim" data-path="$(CURDIR)/data/data_2.bin"'

node3: $(FULL)
	$(SN) --port 9097 --node-config 'partition-id=3 backend="sim" data-path="$(CURDIR)/data/data_3.bin"'

board-reset: $(CORE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.board_stats reset --port $(PORT)

flwr-run: $(FULL)
	$(SN_BIN) flwr run host lab2 --stream --run-config "$(RUNCFG)"

board-read: $(CORE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.board_stats read --port $(PORT) --out ../results/$(EXP)/board_stats.json

board-global: $(CORE)
	cd host && PYTHONPATH=. $(VPY) -m fl_lab.board_global --port $(PORT) --weights ../$(WEIGHTS) \
	  --test ../data/test.bin --out ../$(dir $(WEIGHTS))board_global.json

check-server:
	@grep -nwiE 'serial|port|backend|board|protocol|transports' host/fl_lab/server_app.py host/fl_lab/serverlib.py \
	  && echo "ЗНАЙДЕНО: сервер знає про плату" || echo "(порожньо): серверний код не згадує плату"

lab2-table: $(CORE)
	$(VPY) scripts/lab2_table.py results/exp-a results/exp-b results/exp-c results/exp-g

lab2-plot: $(CORE)
	$(VPY) scripts/plot_lab2.py results results/exp-a results/exp-b results/exp-g
