#!/bin/bash

for i in $(seq 1 10); do
    for gpus in 7 6 5 4 3 2 1; do
        echo "Evaluation $i on $gpus GPU(s)"
        date

        env CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpus - 1))) \
        time torchrun --standalone --nproc_per_node="$gpus" \
            ddp_evaluate_graph_transformer.py \
            tii-ssrc-23-multiclass \
            --split all \
            --batch-size 512 \
            --lap-pe-backend torch \
	    --lap-pe-sign-flip random \
            --throughput \
            --include-transfer-in-throughput

        status=$?
        date
	mv Results/Pickle/tii-ssrc-23-multiclass-throughput-all.pkl Results/Pickle/tii-ssrc-23-multiclass-throughput-all-"$gpus"-"$i".pkl 

        if [ "$status" -ne 0 ]; then
            echo "Evaluation $i on $gpus GPU(s) failed with exit code $status, continuing..."
        fi

        echo
    done
done
