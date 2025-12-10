export BASE_PORT=${BASE_PORT:-13181}
export ASCENT_PYTHON=${ASCENT_PYTHON:-`which python`}

# export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-pretrained_weights/mobile_sam.pt}
# export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
# export GROUNDING_DINO_WEIGHTS=${pretrained_weights/groundingdino_swint_ogc.pth}
# export RAM_CHECKPOINT=${RAM_CHECKPOINT:-pretrained_weights/ram_plus_swin_large_14m.pth}

export QWEN2_5_PORT=${QWEN2_5_PORT:-$((BASE_PORT))}
export BLIP2ITM_PORT=${BLIP2ITM_PORT:-$((BASE_PORT + 1))}
export SAM_PORT=${SAM_PORT:-$((BASE_PORT + 2))}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-$((BASE_PORT + 3))}
export RAM_PORT=${RAM_PORT:-$((BASE_PORT + 4))}
export DFINE_PORT=$((BASE_PORT + 5))

# export CURL_CA_BUNDLE=""
# export HF_ENDPOINT="https://hf-mirror.com"

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

tmux split-window -v -t ${session_name}:0
tmux split-window -v -t ${session_name}:0

tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2
tmux split-window -h -t ${session_name}:0.4

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 "export CUDA_VISIBLE_DEVICES=0; ${ASCENT_PYTHON} -m model_api.qwen25_out --port ${QWEN2_5_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "export CUDA_VISIBLE_DEVICES=1; ${ASCENT_PYTHON} -m model_api.blip2itm_out --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "export CUDA_VISIBLE_DEVICES=1; ${ASCENT_PYTHON} -m model_api.sam_out --port ${SAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "export CUDA_VISIBLE_DEVICES=1; ${ASCENT_PYTHON} -m model_api.grounding_dino_out --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.4 "export CUDA_VISIBLE_DEVICES=1; ${ASCENT_PYTHON} -m model_api.ram_out --port ${RAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.5 "export CUDA_VISIBLE_DEVICES=1; ${ASCENT_PYTHON} -m model_api.dfine_out --port ${DFINE_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"
echo "tmux kill-session -t ${session_name}"

