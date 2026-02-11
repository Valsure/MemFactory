
import os

# Define models
models = {
    "1.7B": "/home/models/Qwen3-1.7B",
    "4B": "/home/models/qwen3-4b",
    # Placeholder for 3B as requested, though not found on disk
    "3B": "/home/models/Qwen2.5-3B" 
}

# Define configurations
configs = {
    "baseline": {"beta": 0.1, "accum": 1, "lr": 5e-7},
    "recommended": {"beta": 0.01, "accum": 4, "lr": 1e-6},
    "aggressive": {"beta": 0.01, "accum": 8, "lr": 2e-6}
}

# Training modes
modes = {
    "all": "",
    "ext_only": "--no_train_update",
    "upd_only": "--no_train_extraction"
}

# Base directory for scripts
output_dir = "."

# Generate scripts
for m_name, m_path in models.items():
    for c_name, c_params in configs.items():
        for mode_name, mode_arg in modes.items():
            exp_name = f"{m_name}_{c_name}_{mode_name}"
            filename = os.path.join(output_dir, f"run_{exp_name}.sh")
            
            # Prepare the script content
            # Note: We assume the script is run from the 'RL' directory or 'RL/experiments' directory?
            # Let's assume user runs from 'RL/experiments' so we need to go up one level for python script.
            # But user might run from 'RL'.
            # Let's make it robust by assuming user runs ./experiments/script.sh from RL, or cd experiments && ./script.sh
            # Safest is to use absolute paths or relative to script location.
            # I'll assume running from RL root is standard, but these scripts are in subfolder.
            # Let's use `cd ..` if running from inside experiments.
            
            # Actually, let's just make them simple and assume running from RL directory:
            # "usage: ./experiments/run_..."
            
            content = f"""#!/bin/bash

# Experiment: {exp_name}
# Model: {m_name} ({m_path})
# Config: {c_name} (Beta: {c_params['beta']}, Accum: {c_params['accum']}, LR: {c_params['lr']})
# Mode: {mode_name}

# Ensure we are in the RL directory (parent of experiments)
cd "$(dirname "$0")/.." || exit

echo "Starting training for {exp_name}..."

python RL/mem_grpo_trainer.py \\
    --model_name_or_path "{m_path}" \\
    --output_dir "./output/{exp_name}" \\
    --data_path "./datas/train.jsonl" \\
    --beta {c_params['beta']} \\
    --gradient_accumulation_steps {c_params['accum']} \\
    --lr {c_params['lr']} \\
    --save_steps 9999 \\
    --max_generate_length 4096 \\
    --epoch 2 \\
    --wandb_name "{exp_name}" \\
    {mode_arg}

echo "Training finished for {exp_name}"
"""
            
            if m_name == "3B":
                content = "# WARNING: The model path below is a placeholder and may not exist.\n" + content

            with open(filename, "w") as f:
                f.write(content)
            
            # Make executable
            os.chmod(filename, 0o755)
            print(f"Created {filename}")

print("All scripts created.")
