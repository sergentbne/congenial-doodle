#!/usr/bin/env bash
SESSION="inference_model"  # session name

# if session exists, attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

# create new session detached
tmux new-session -d -s "$SESSION" -n "inference testing"

tmux split-window -v -t "$SESSION:0"
tmux split-window -h -t "$SESSION:0.1"
# window 1: editor
tmux send-keys -t "$SESSION:0.0" "cd /home/lrima/Documents/prog/inference/Metric3D" C-m
tmux send-keys -t "$SESSION:0.1" "cd /home/lrima/Documents/prog/inference/Metric3D" C-m
tmux send-keys -t "$SESSION:0.2" "cd /home/lrima/Documents/prog/inference/Metric3D" C-m

#update system
tmux send-keys -t "$SESSION:0.0" "git pull" C-m

#launch system
tmux send-keys -t "$SESSION:0.0" "uv run inference_v3.py /mnt/harddrive/training_data/nuscenes_data /mnt/harddrive/training_data" C-m
tmux send-keys -t "$SESSION:0.2" "glances" C-m
tmux send-keys -t "$SESSION:0.1" "nvtop" C-m

# window 2: server + logs (split vertically)
tmux new-window -t "$SESSION" -n "shell"
tmux send-keys -t "$SESSION:1" "cd /home/lrima/Documents/prog/inference/Metric3D" C-m

tmux select-window -t "$SESSION:0"

tmux attach -t "$SESSION"

