GRAPH=./output/output.jpeg
DATA=./data/time_per_core.dat
SORTED_DATA=./data/time_per_core_sorted.dat

run: $(DATA)
	make graph

graph: $(GRAPH)

sort: $(SORTED_DATA)

test:
	make clean
	bash scoop.sh
	#SLURM_ARRAY_TASK_ID=1 bash scoop.sh
	#make graph

clean:
	rm -f scoop.log
	rm -f scoop.lock
	rm -f $(SORTED_DATA)
	rm -f $(GRAPH)
	rm -f *.pyc
	rm -f $(DATA)


$(GRAPH): scoop.gpi
	make sort
	gnuplot scoop.gpi

$(SORTED_DATA): $(DATA)
	sort -V $(DATA) > $(SORTED_DATA)

$(DATA): ./scoop.sh ./pi_calc.py
	make clean
	sbatch -W scoop.sh

