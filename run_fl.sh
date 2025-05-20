#!/bin/bash

# Script to run federated learning experiments using Flower 1.18.0+
# This script handles dependency installation, data preprocessing, and experiment execution

export PYTHONPATH=$(pwd)

# Function to display help message
show_help() {
    echo "Federated Learning Experiments Script (Flower 1.18.0+)"
    echo ""
    echo "Usage: ./run_fl.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help                      Show this help message"
    echo "  --install                   Install dependencies"
    echo "  --preprocess                Preprocess data"
    echo "  --exp TYPE                  Experiment type: 'fedavg', 'mtfl', or 'both'"
    echo "  --rounds NUM                Maximum number of federated learning rounds (default: 500)"
    echo "  --target-accuracy NUM       Target accuracy for termination (e.g., 0.65 for 65%)"
    echo "  --save-init                 Save initial model weights"
    echo "  --local-epochs NUM          Number of local training epochs (default: 1)"
    echo "  --client-fraction NUM       Fraction of clients to sample each round (default: 0.1)"
    echo "  --num-clients NUM           Total number of clients (default: 20)"
    echo "  --port NUM                  Port to use for server (default: 8080)"
    echo "  --visualize                 Visualize results after experiment"
    echo "  --seed NUM                  Random seed for reproducibility (default: 42)"
    echo "  --learning-rate NUM         Learning rate for training (default: 0.01)"
    echo "  --weights-dir DIR           Directory to store model weights"
    echo "  --metrics-dir DIR           Directory to store metrics"
    echo "  --partitions-dir DIR        Directory to store data partitions"
    echo "  --bn-private TYPE           Private BN type: 'none', 'gamma_beta', 'mu_sigma', 'all'"
    echo "  --optimizer TYPE            Optimizer type: 'sgd', 'adam'"
    echo "  --beta1 VALUE               Beta1 value for Adam optimizer (default: 0.9)"
    echo "  --beta2 VALUE               Beta2 value for Adam optimizer (default: 0.999)"
    echo "  --epsilon VALUE             Epsilon value for Adam optimizer (default: 1e-8)"
    echo "  --log-wandb                 Enable logging to Weights & Biases"
    echo "  --project-name NAME         W&B project name (default: 'mtfl-replication')"
    echo "  --experiment-tag TAG        Tag for the experiment (e.g., 'result1')"
    echo "  --per-step-metrics          Log metrics after each local step (for Result 2)"
    echo "  --steps-per-epoch NUM       Number of steps per local epoch (default: 10)"
    echo "  --checkpoint-interval NUM   Interval (in rounds) to save checkpoints (default: 0 = disabled)"
    echo ""
    echo "Examples:"
    echo "  ./run_fl.sh --install                  # Only install dependencies"
    echo "  ./run_fl.sh --preprocess --num-clients 400  # Only preprocess data"
    echo "  ./run_fl.sh --exp fedavg --rounds 200 --target-accuracy 0.65  # Run FedAvg experiment"
    echo "  ./run_fl.sh --exp both --visualize --bn-private gamma_beta    # Run both experiments and visualize"
    echo "  ./run_fl.sh --exp mtfl --checkpoint-interval 10               # Save model checkpoints every 10 rounds"
    echo ""
    echo "Note: If --install or --preprocess are used without --exp, only those tasks will be performed."
    echo "Note: Use --rounds, not --max-rounds, to set the number of training rounds."
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if server is ready
wait_for_server() {
    local port=$1
    echo "Waiting for server to be ready on port $port..."
    
    # Use netcat if available, otherwise use basic retry logic
    if command_exists nc; then
        for i in {1..30}; do
            if nc -z localhost $port 2>/dev/null; then
                echo "Server is running on port $port"
                return 0
            fi
            sleep 1
            echo -n "."
            if [ $i -eq 30 ]; then
                echo ""
                echo "Warning: Server didn't respond within 30 seconds, starting clients anyway"
            fi
        done
    else
        # Basic retry logic if nc not available
        for i in {1..10}; do
            sleep 3
            echo -n "."
        done
        echo ""
        echo "Assuming server is up after 30s delay"
    fi
}

# Function to get Flower version
get_flower_version() {
    # Get the installed Flower version
    FLOWER_VERSION=$(pip show flwr 2>/dev/null | grep "Version" | awk '{print $2}')
    echo $FLOWER_VERSION
}

# Default values
EXP_TYPE=""  # Empty means no experiment will be run
MAX_ROUNDS=500
TARGET_ACCURACY=""
SAVE_INIT=false
LOCAL_EPOCHS=1
CLIENT_FRACTION=0.1
NUM_CLIENTS=20
PORT=8080
INSTALL=false
PREPROCESS=false
VISUALIZE=false
RUN_EXPERIMENT=false  # Flag to determine if we should run an experiment
SEED=42
LEARNING_RATE=0.01
WEIGHTS_DIR="./data/weights"
METRICS_DIR="./metrics"
PARTITIONS_DIR="./data/partitions"
PARTITIONS_READY=false
BN_PRIVATE="none"
OPTIMIZER="sgd"
BETA1=0.9
BETA2=0.999
EPSILON=1e-8
LOG_WANDB=false
PROJECT_NAME="mtfl-replication"
EXPERIMENT_TAG=""
PER_STEP_METRICS=false
STEPS_PER_EPOCH=10
CHECKPOINT_INTERVAL=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --preprocess)
            PREPROCESS=true
            shift
            ;;
        --exp)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                EXP_TYPE="$2"
                RUN_EXPERIMENT=true
                shift 2
            else
                echo "Error: --exp requires an argument."
                exit 1
            fi
            ;;
        --rounds)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                MAX_ROUNDS="$2"
                shift 2
            else
                echo "Error: --rounds requires an argument."
                exit 1
            fi
            ;;
        --max-rounds)
            echo "Error: --max-rounds is not a valid option. Please use --rounds instead."
            exit 1
            ;;
        --target-accuracy)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                TARGET_ACCURACY="$2"
                shift 2
            else
                echo "Error: --target-accuracy requires an argument."
                exit 1
            fi
            ;;
        --save-init)
            SAVE_INIT=true
            shift
            ;;
        --local-epochs)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                LOCAL_EPOCHS="$2"
                shift 2
            else
                echo "Error: --local-epochs requires an argument."
                exit 1
            fi
            ;;
        --client-fraction)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                CLIENT_FRACTION="$2"
                shift 2
            else
                echo "Error: --client-fraction requires an argument."
                exit 1
            fi
            ;;
        --num-clients)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                NUM_CLIENTS="$2"
                shift 2
            else
                echo "Error: --num-clients requires an argument."
                exit 1
            fi
            ;;
        --port)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                PORT="$2"
                shift 2
            else
                echo "Error: --port requires an argument."
                exit 1
            fi
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        --seed)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                SEED="$2"
                shift 2
            else
                echo "Error: --seed requires an argument."
                exit 1
            fi
            ;;
        --learning-rate)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                LEARNING_RATE="$2"
                shift 2
            else
                echo "Error: --learning-rate requires an argument."
                exit 1
            fi
            ;;
        --weights-dir)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                WEIGHTS_DIR="$2"
                shift 2
            else
                echo "Error: --weights-dir requires an argument."
                exit 1
            fi
            ;;
        --metrics-dir)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                METRICS_DIR="$2"
                shift 2
            else
                echo "Error: --metrics-dir requires an argument."
                exit 1
            fi
            ;;
        --partitions-dir)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                PARTITIONS_DIR="$2"
                shift 2
            else
                echo "Error: --partitions-dir requires an argument."
                exit 1
            fi
            ;;
        --bn-private)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                BN_PRIVATE="$2"
                shift 2
            else
                echo "Error: --bn-private requires an argument."
                exit 1
            fi
            ;;
        --optimizer)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                OPTIMIZER="$2"
                shift 2
            else
                echo "Error: --optimizer requires an argument."
                exit 1
            fi
            ;;
        --beta1)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                BETA1="$2"
                shift 2
            else
                echo "Error: --beta1 requires an argument."
                exit 1
            fi
            ;;
        --beta2)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                BETA2="$2"
                shift 2
            else
                echo "Error: --beta2 requires an argument."
                exit 1
            fi
            ;;
        --epsilon)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                EPSILON="$2"
                shift 2
            else
                echo "Error: --epsilon requires an argument."
                exit 1
            fi
            ;;
        --log-wandb)
            LOG_WANDB=true
            shift
            ;;
        --project-name)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                PROJECT_NAME="$2"
                shift 2
            else
                echo "Error: --project-name requires an argument."
                exit 1
            fi
            ;;
        --experiment-tag)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                EXPERIMENT_TAG="$2"
                shift 2
            else
                echo "Error: --experiment-tag requires an argument."
                exit 1
            fi
            ;;
        --per-step-metrics)
            PER_STEP_METRICS=true
            shift
            ;;
        --steps-per-epoch)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                STEPS_PER_EPOCH="$2"
                shift 2
            else
                echo "Error: --steps-per-epoch requires an argument."
                exit 1
            fi
            ;;
        --checkpoint-interval)
            if [[ -n "$2" ]] && [[ "$2" != --* ]]; then
                CHECKPOINT_INTERVAL="$2"
                shift 2
            else
                echo "Error: --checkpoint-interval requires an argument."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p "${WEIGHTS_DIR}" "${METRICS_DIR}" "./figures" "${PARTITIONS_DIR}" "./data"

# Check if only install and/or preprocess are requested (without experiment)
if [ "$INSTALL" = true ] || [ "$PREPROCESS" = true ]; then
    if [ "$RUN_EXPERIMENT" = false ]; then
        # Only perform installation and/or preprocessing without running experiments
        
        if [ "$INSTALL" = true ]; then
            echo "Installing dependencies..."
            pip install "flwr>=1.18.0" torch torchvision datasets matplotlib numpy wandb
            echo "Dependencies installed successfully."
        fi
        
        if [ "$PREPROCESS" = true ]; then
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            echo "Starting preprocessing at: $TIMESTAMP"
            
            python preprocess_data.py --num-clients "$NUM_CLIENTS" --seed "$SEED" \
                                    --partitions-dir "$PARTITIONS_DIR"
            
            # Check if preprocessing was successful
            if [ $? -ne 0 ]; then
                echo "Error: Data preprocessing failed."
                exit 1
            fi
            
            echo "Data preprocessing completed successfully at $(date +"%Y-%m-%d %H:%M:%S")"
            
            # Set a flag to inform experiment that partitions are already created
            PARTITIONS_READY=true
        fi
        
        echo "Completed requested tasks. No experiments will be run (--exp was not specified)."
        exit 0
    else
        # Both installation/preprocessing AND experiment are requested
        # Perform installation first if requested
        if [ "$INSTALL" = true ]; then
            echo "Installing dependencies..."
            pip install "flwr>=1.18.0" torch torchvision datasets matplotlib numpy wandb
            echo "Dependencies installed successfully."
        fi
        
        # Then preprocess if requested
        if [ "$PREPROCESS" = true ]; then
            echo "Preprocessing data..."
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            echo "Starting preprocessing at: $TIMESTAMP"
            
            python preprocess_data.py --num-clients "$NUM_CLIENTS" --seed "$SEED" \
                                    --partitions-dir "$PARTITIONS_DIR"
            
            # Check if preprocessing was successful
            if [ $? -ne 0 ]; then
                echo "Error: Data preprocessing failed."
                exit 1
            fi
            
            echo "Data preprocessing completed successfully at $(date +"%Y-%m-%d %H:%M:%S")"
            
            # Set a flag to inform experiment that partitions are already created
            PARTITIONS_READY=true
        fi
        
        # Continue to experiment setup below...
    fi
else
    # Neither install nor preprocess was requested, but check if experiment was requested
    if [ "$RUN_EXPERIMENT" = false ]; then
        echo "No tasks specified. Use --install, --preprocess, or --exp to specify a task."
        echo "Use --help for more information."
        exit 0
    fi
    # Continue to experiment setup below...
fi

# Get Flower version
FLOWER_VERSION=$(get_flower_version)
echo "Using Flower version: $FLOWER_VERSION"

# Verify minimum Flower version
if [[ "$(printf '%s\n' "1.18.0" "$FLOWER_VERSION" | sort -V | head -n1)" != "1.18.0" ]]; then
    echo "Error: Flower version must be at least 1.18.0. Please upgrade with: pip install 'flwr>=1.18.0'"
    exit 1
fi

# Set up cleanup on exit
trap 'kill $(jobs -p) 2>/dev/null; echo "Cleaning up processes..."; wait' EXIT

# Function to run a single experiment
run_experiment() {
    local exp_type=$1
    local port=$2
    local save_init_flag=""
    local start_unix=$(date +%s)
    local timestamp_str=$(date +"%Y-%m-%d %H:%M:%S")
    
    if [ "$SAVE_INIT" = true ]; then
        save_init_flag="--save-init"
    fi
    
    echo ""
    echo "======================================================="
    echo "Starting $exp_type experiment with $NUM_CLIENTS clients and $MAX_ROUNDS max rounds"
    if [ -n "$TARGET_ACCURACY" ]; then
        echo "Target accuracy: $TARGET_ACCURACY"
    fi
    if [ "$CHECKPOINT_INTERVAL" -gt 0 ]; then
        echo "Checkpoint interval: $CHECKPOINT_INTERVAL rounds"
    fi
    echo "Started at: $timestamp_str"
    echo "======================================================="
    echo ""    
    
    # Check if port is already in use
    if command_exists lsof; then
        lsof -i:$port 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "WARNING: Port $port is already in use. This may cause conflicts."
        fi
    fi
    
    # Start the server
    echo "Starting server on port $port..."
    python fl_mtfl/server_app.py \
        --exp "$exp_type" \
        --rounds "$MAX_ROUNDS" \
        --local-epochs "$LOCAL_EPOCHS" \
        --client-fraction "$CLIENT_FRACTION" \
        --num-clients "$NUM_CLIENTS" \
        --seed "$SEED" \
        --learning-rate "$LEARNING_RATE" \
        --weights-dir "$WEIGHTS_DIR" \
        --metrics-dir "$METRICS_DIR" \
        --partitions-dir "$PARTITIONS_DIR" \
        --bn-private "$BN_PRIVATE" \
        --optimizer "$OPTIMIZER" \
        --beta1 "$BETA1" \
        --beta2 "$BETA2" \
        --epsilon "$EPSILON" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        ${TARGET_ACCURACY:+--target-accuracy "$TARGET_ACCURACY"} \
        ${LOG_WANDB:+--log-wandb} \
        --project-name "$PROJECT_NAME" \
        ${EXPERIMENT_TAG:+--experiment-tag "$EXPERIMENT_TAG"} \
        ${PER_STEP_METRICS:+--per-step-metrics} \
        --steps-per-epoch "$STEPS_PER_EPOCH" \
        $save_init_flag \
        --server-address "localhost:$port" &
    
    server_pid=$!
    echo "Server started with PID: $server_pid"
    
    # Wait for server to be ready before starting clients
    wait_for_server $port
    
    # Start clients using Flower's Virtual Client Engine
    echo "Starting $NUM_CLIENTS clients..."
    
    # Create temporary Python script to run clients
    tmp_client_script=$(mktemp)
    cat > "$tmp_client_script" << EOF
import sys
import os
from fl_mtfl.client_app import client_app

# Set environment variable for client ID
os.environ["FLOWER_CLIENT_ID"] = sys.argv[1]

# Properly reset sys.argv for argparse (simulate normal script call)
sys.argv = [sys.argv[0]] + sys.argv[2:]

# Launch the client app
client_app()
EOF
    
    # Start clients in the background
    client_pids=()
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
        FLOWER_CLIENT_ID=$i python "$tmp_client_script" "$i" \
            --exp "$exp_type" \
            --rounds "$MAX_ROUNDS" \
            --local-epochs "$LOCAL_EPOCHS" \
            --client-fraction "$CLIENT_FRACTION" \
            --num-clients "$NUM_CLIENTS" \
            --seed "$SEED" \
            --learning-rate "$LEARNING_RATE" \
            --weights-dir "$WEIGHTS_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --partitions-dir "$PARTITIONS_DIR" \
            --bn-private "$BN_PRIVATE" \
            --optimizer "$OPTIMIZER" \
            --beta1 "$BETA1" \
            --beta2 "$BETA2" \
            --epsilon "$EPSILON" \
            ${TARGET_ACCURACY:+--target-accuracy "$TARGET_ACCURACY"} \
            ${LOG_WANDB:+--log-wandb} \
            --project-name "$PROJECT_NAME" \
            ${EXPERIMENT_TAG:+--experiment-tag "$EXPERIMENT_TAG"} \
            ${PER_STEP_METRICS:+--per-step-metrics} \
            --steps-per-epoch "$STEPS_PER_EPOCH" \
            --server-address "localhost:$port" &
    
        pid=$!
        client_pids+=($pid)
        echo "Started client $i with PID: $pid"
    
        # Stagger client start to avoid overwhelming the server
        sleep 0.2
    done
    
    # Wait for the server to finish
    echo "Waiting for server to complete..."
    wait $server_pid || true
    server_exit_code=$?
    
    # Clean up the temporary client script
    rm "$tmp_client_script"
    
    # Check if server completed successfully
    local end_unix=$(date +%s)
    if [ $server_exit_code -ne 0 ]; then
        echo "Server process failed with exit code $server_exit_code"
    else
        echo "$exp_type experiment completed successfully at $(date +"%Y-%m-%d %H:%M:%S")"
        echo "Experiment duration: $((end_unix - start_unix)) seconds"
    fi
    
    # Terminate all clients
    echo "Terminating clients..."
    for pid in "${client_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null
        fi
    done
    
    # Ensure the server is terminated if it's still running
    if kill -0 $server_pid 2>/dev/null; then
        kill $server_pid 2>/dev/null
        echo "Terminated server (PID: $server_pid)"
    fi
    
    # Wait a moment to ensure all processes are terminated
    sleep 1
    
    # Return server exit code
    return $server_exit_code
}

# Run experiments based on type
if [ "$EXP_TYPE" = "both" ]; then
    run_experiment "fedavg" $PORT
    fedavg_exit_code=$?
    
    # Increment port for next experiment
    PORT=$((PORT + 1))
    
    run_experiment "mtfl" $PORT
    mtfl_exit_code=$?
    
    # Create comparative metrics
    echo "Creating comparative metrics..."
    python -c "
import sys
sys.path.append('.')
from fl_mtfl.server_app import create_comparative_metrics
create_comparative_metrics('$METRICS_DIR')
"
    
    # Overall success depends on both experiments
    if [ $fedavg_exit_code -ne 0 ] || [ $mtfl_exit_code -ne 0 ]; then
        echo "WARNING: At least one experiment did not complete successfully."
    fi
    
elif [ "$EXP_TYPE" = "fedavg" ] || [ "$EXP_TYPE" = "mtfl" ]; then
    run_experiment "$EXP_TYPE" $PORT
    experiment_exit_code=$?
    
    if [ $experiment_exit_code -ne 0 ]; then
        echo "Experiment did not complete successfully."
    fi
else
    echo "Invalid experiment type: $EXP_TYPE. Must be 'fedavg', 'mtfl', or 'both'."
    exit 1
fi

# Visualize results if requested
if [ "$VISUALIZE" = true ] && [ "$RUN_EXPERIMENT" = true ]; then
    echo "Creating visualizations..."
    if [ "$EXP_TYPE" = "both" ]; then
        python visualize_results.py --metrics-dir "$METRICS_DIR" --figures-dir "./figures"
    elif [ "$EXP_TYPE" = "fedavg" ]; then
        python visualize_results.py --metrics-dir "$METRICS_DIR" --figures-dir "./figures" --fedavg-only
    elif [ "$EXP_TYPE" = "mtfl" ]; then
        python visualize_results.py --metrics-dir "$METRICS_DIR" --figures-dir "./figures" --mtfl-only
    fi
    
    echo "Visualizations saved to ./figures directory."
fi

echo "All tasks completed."