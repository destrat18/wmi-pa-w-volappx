

bash -c "cd /project/t3_sfarokhnia/wmi-pa-w-volappx && GRB_LICENSE_FILE=/project/t3_sfarokhnia/wmi-pa-w-volappx/gurobi1.lic "


for i in $(seq 0 29); do timeout 3600 python3 experiments.py --benchmark rational_2 --benchmark-path experimental_results/random_benchmarks_rational_2.json --faza --epsilon 0.1 --max-workers 16 --benchmark-index $i; done;


for i in $(seq 0 3); do timeout 3600 python3 experiments.py --benchmark manual --faza --epsilon 0.1 --max-workers 16 --benchmark-index $i; done;