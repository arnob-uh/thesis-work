#!/bin/bash

echo "Starting all dataset experiments..."

# List of dataset scripts
# SCRIPTS=("ETTh1.sh" "ETTm1.sh" "electricity.sh" "weather.sh")
SCRIPTS=("weather.sh")

# Loop through and run each script
for script in "${SCRIPTS[@]}"; do
    echo "Running $script..."
    bash "$script"
    echo "$script completed."
done

echo "All experiments completed."