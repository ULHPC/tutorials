FITNESS=fitness_per_tries
TIME=time_per_tries
DATA_DIR=./data
GRAPH_DIR=./output

DATA_TIME=$(DATA_DIR)/$(TIME).dat
DATA_FITNESS=$(DATA_DIR)/$(FITNESS).dat
SORTED_DATA_TIME=$(DATA_DIR)/$(TIME)_sorted.dat
SORTED_DATA_FITNESS=$(DATA_DIR)/$(FITNESS)_sorted.dat
GRAPH_TIME=$(GRAPH_DIR)/$(TIME).jpeg
GRAPH_FITNESS=$(GRAPH_DIR)/$(FITNESS).jpeg

run:
	make graph

graph: $(GRAPH_FITNESS) $(GRAPH_TIME)

sort: $(SORTED_DATA_FITNESS) $(SORTED_DATA_TIME)

getdata: ./pi_calc.sh ./pi_calc.py
	sbatch pi_calc.sh

test:
	make clean
	for i in 100 1000 10000 100000 1000000 10000000 100000000; do SLURM_ARRAY_TASK_ID=$$i bash pi_calc.sh; done;
	make graph

clean:
	rm -f pi_calc.log
	rm -f pi_calc.lock
	rm -f data/*.dat
	rm -f output/*.jpeg
	rm -f *.pyc


$(GRAPH_FITNESS): $(FITNESS).gpi
	make sort
	gnuplot $(FITNESS).gpi

$(GRAPH_TIME): $(TIME).gpi
	make sort
	gnuplot $(TIME).gpi

$(SORTED_DATA_TIME): $(DATA_TIME)
	sort -V $(DATA_TIME) > $(SORTED_DATA_TIME)

$(SORTED_DATA_FITNESS): $(DATA_FITNESS)
	sort -V $(DATA_FITNESS) > $(SORTED_DATA_FITNESS)

$(DATA_TIME):
	make getdata

$(DATA_FITNESS):
	make getdata

