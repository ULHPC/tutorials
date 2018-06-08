GRAPH=./output/output.jpeg
DATA=./data/time_per_core.dat
SORTED_DATA=./data/time_per_core_sorted.dat
SPEEDUP_DATA=./data/time_per_core_speedup.dat

run: $(DATA)
	make graph

graph: $(GRAPH)

sort: $(SORTED_DATA)

compute_speedup: $(SPEEDUP_DATA)

test:
	make clean
	SLURM_ARRAY_TASK_ID=2 bash scoop.sh
	SLURM_ARRAY_TASK_ID=1 bash scoop.sh
	make graph

clean:
	rm -f scoop.log
	rm -f scoop.lock
	rm -f $(SORTED_DATA)
	rm -f $(GRAPH)
	rm -f *.pyc
	rm -f $(DATA)


$(GRAPH): scoop.gpi
	make sort
	make compute_speedup
	gnuplot gnuplot/scoop.gpi

$(SORTED_DATA): $(DATA)
	sort -V $(DATA) > $(SORTED_DATA)

$(SPEEDUP_DATA): $(SORTED_DATA)
	awk -f gnuplot/compute_speedup.awk $(SORTED_DATA)

$(DATA): ./scoop.sh ./pi_calc.py
	make clean
	sbatch -W scoop.sh

