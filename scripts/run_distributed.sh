#!/bin/bash

# This script launches a 3-worker distributed training job.

# The script is now designed to be run from the project root.

PEERS="localhost:50051,localhost:50052,localhost:50053"
TIMEOUT=10s

echo "Starting worker 0..."
go run ./cmd/distributed/main.go --rank=0 --peers=$PEERS --timeout=$TIMEOUT > worker0.log 2>&1 &
P1=$!

sleep 3

echo "Starting worker 1..."
go run ./cmd/distributed/main.go --rank=1 --peers=$PEERS --timeout=$TIMEOUT > worker1.log 2>&1 &
P2=$!

sleep 3

echo "Starting worker 2..."
go run ./cmd/distributed/main.go --rank=2 --peers=$PEERS --timeout=$TIMEOUT > worker2.log 2>&1 &
P3=$!

echo "Waiting for workers to finish..."
wait $P1
wait $P2
wait $P3

echo "Distributed training complete."
