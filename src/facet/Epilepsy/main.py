import sys
import os

# ---import sys
import os

# --- Setup Python Path ---
# This ensures the script can find your 'facet' module
# Adjust the path if your project structure is different
project_root = r"D:\Medical Engineering and Analytics\Project\FACETpy"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
# -------------------------

from facet.Epilepsy.pipeline import run_combined_pipeline
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the EEG-fMRI analysis pipeline.
    """
    # 1. Define the path to your .mat file
    file_name = "DA00100S.mat"
    mat_file_path = os.path.join(project_root, "examples", "datasets", "MAT_Files", file_name)

    print(f"Starting analysis for: {mat_file_path}")

    # 2. Set the parameters for the pipeline
    # to generate the fMRI regressors.
    pipeline_params = {
        "mat_path": mat_file_path,
        "sfreq": 500.0,      # The sampling frequency of your EEG data
        "has_fmri": True,    # Set to True to generate fMRI regressors
        "tr": 2.5,           # The Repetition Time (TR) of fMRI data
        "visualize": False,  # Set to True if you want to see ICA plots during runtime
    }

    # 3. Run the combined pipeline
    results = run_combined_pipeline(**pipeline_params)
    print("Pipeline execution finished.")

    # 4. Inspect the results
    if results and results.get("detection"):
        print("\n--- Analysis Summary ---")
        
        detection_result = results["detection"]
        print(f"Number of accepted ICA components: {len(detection_result.accepted_components)}")
        if len(detection_result.accepted_components) > 0:
            print(f"Accepted Component Indices: {detection_result.accepted_components}")
            print("Tip: You can visualize these components using 'detection_result.ica.plot_components(picks=detection_result.accepted_components)'")
            
            try:
                import matplotlib.pyplot as plt
                # Check if we can plot (requires montage)
                if detection_result.ica.info.get_montage() is None:
                    print("Warning: No standard montage (digitization points) found. Skipping topography plot to avoid crash.")
                    print("To fix: Ensure channel names match standard 10-20 (e.g., Fp1, C3) and set montage in preprocessing.")
                else:
                    print("Plotting accepted component topographies...")
                    detection_result.ica.plot_components(picks=detection_result.accepted_components)
                    plt.show()
            except Exception as e:
                print(f"Could not plot components: {e}")

        print(f"Number of refined spike times: {len(detection_result.refined_times)}")

        # Check for the generated regressors
        reg_g = None
        reg_e = None

        if results.get("regressor_grouiller"):
            reg_g = results["regressor_grouiller"]["regressor_hrf"]
            print(f"Grouiller (event-based) regressor shape:    {reg_g.shape}")

        if results.get("regressor_ebrahimzadeh") is not None:
            reg_e = results["regressor_ebrahimzadeh"]
            print(f"Ebrahimzadeh (ICA-based) regressor shape:   {reg_e.shape}")
        
        # Compare shapes
        if reg_g is not None and reg_e is not None:
            if reg_g.shape == reg_e.shape:
                print(" SUCCESS: Regressor shapes match.")
            else:
                print(f" WARNING: Shape mismatch! ({reg_g.shape} vs {reg_e.shape})")
        
        print("\nResults dictionary contains 'detection', 'regressor_grouiller', and 'regressor_ebrahimzadeh'.")
    else:
        print("The pipeline did not return any results. Please check the logs for warnings or errors.")


if __name__ == "__main__":
    main()