SYN_DIR=synthetic_exp

mkdir -p $SYN_DIR/plots

python3 plotUAI.py $SYN_DIR/results/* -o $SYN_DIR/plots/ -f _syn_r3_b3_d4-7
python3 plotUAI.py $SYN_DIR/results/* -o $SYN_DIR/plots/ -f _syn_r3_b3_d4-7_cactus --cactus
# python3 plotUAI_cactus.py $SYN_DIR/results/* -o $SYN_DIR/plots/ -f _syn_r3_b3_d4-7_cactus
