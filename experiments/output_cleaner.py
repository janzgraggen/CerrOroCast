import os
import shutil

def clean_output_directory(output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        print(f"Directory '{output_dir}' does not exist.")
        return

    # Iterate through subdirectories in the output directory
    for model_dir in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model_dir)

        # Check if the model_path is a directory
        if not os.path.isdir(model_path) or "SIDM" in model_dir:
            continue

        # Iterate through subdirectories of the model directory
        for run_dir in os.listdir(model_path):
            run_path = os.path.join(model_path, run_dir)

            # Check if the run_path is a directory
            if not os.path.isdir(run_path):
                continue

            # Check if the 'checkpoints' subdirectory exists
            checkpoints_path = os.path.join(run_path, 'checkpoints')
            if not os.path.exists(checkpoints_path) or not os.path.isdir(checkpoints_path):
                # Delete the run directory if 'checkpoints' does not exist
                print(f"Deleting run directory: {run_path}")
                shutil.rmtree(run_path)

if __name__ == "__main__":
    output_directory = "outputs"  # Replace with the path to your output directory
    clean_output_directory(output_directory)