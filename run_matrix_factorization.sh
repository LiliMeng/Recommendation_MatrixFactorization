#!/bin/bash

# Create a directory for log files
LOG_DIR="logs"
mkdir -p $LOG_DIR
EVAL_INTERVAL=1

# Define the parameter sets
declare -a K_VALUES=(20) # you can add different values, such as (20, 30)
declare -a STEPS_VALUES=(30)
declare -a ALPHA_VALUES=(0.001)
declare -a BETA_VALUES=(0.01)

# Iterate over all parameter combinations
for K in "${K_VALUES[@]}"
do
    for STEPS in "${STEPS_VALUES[@]}"
    do
        for ALPHA in "${ALPHA_VALUES[@]}"
        do
            for BETA in "${BETA_VALUES[@]}"
            do
                # Create a log file name based on parameters
                LOG_FILENAME="${LOG_DIR}/log_K${K}_steps${STEPS}_alpha${ALPHA}_beta${BETA}.txt"
                
                # Run the Python script with the current parameters
                python3 matrix_factorization.py --K $K --steps $STEPS --alpha $ALPHA --beta $BETA --log_filename $LOG_FILENAME --eval_interval $EVAL_INTERVAL
            done
        done
    done
done

## How to run the code:
## Make the Script Executable: chmod +x run_matrix_factorization.sh
## Running the Bash Script: ./run_matrix_factorization.sh --K 30 --steps 1000 --alpha 0.001 --beta 0.01 --log_filename my_log.txt --val_interval 10
