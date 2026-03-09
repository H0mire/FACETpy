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
from facet.Epilepsy.diagnostic_utils import plot_detection_components
from facet.Epilepsy.preprocessing import parse_spike_times
from scipy.io import loadmat
import argparse
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the EEG-fMRI analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Run FACETpy pipeline on a .mat EEG file.')
    parser.add_argument('--mat-file', '-m', default=None, help='Path to .mat file to analyze')
    parser.add_argument('--visualize', action='store_true', help='Show interactive diagnostic plots')
    parser.add_argument('--save-figs', action='store_true', help='Automatically save diagnostic figures and do not show GUI')
    parser.add_argument('--figs-outdir', default=os.path.join(project_root, 'results', 'figures'), help='Directory to save diagnostic figures')
    parser.add_argument('--plot-topk', type=int, default=None, help='If set, plot only the first K accepted components')
    parser.add_argument('--plot-components', default=None, help='Comma-separated list of accepted component indices to plot (e.g. 0,2,5)')
    parser.add_argument('--threshold', type=float, default=0.85, help='Correlation threshold for component acceptance (default: 0.85)')
    args = parser.parse_args()

    # 1. Define the path to your .mat file
    file_name = "DA00100T.mat"
    if args.mat_file:
        mat_file_path = args.mat_file
    else:
        mat_file_path = os.path.join(project_root, "examples", "datasets", "MAT_Files", file_name)

    print(f"Starting analysis for: {mat_file_path}")
    print(f"Using correlation threshold: {args.threshold}")

    # 2. Set the parameters for the pipeline
    # to generate the fMRI regressors.
    pipeline_params = {
        "mat_path": mat_file_path,
        "sfreq": 500.0,      # The sampling frequency of your EEG data
        "has_fmri": True,    # Set to True to generate fMRI regressors
        "tr": 2.5,           # The Repetition Time (TR) of fMRI data
        "visualize": bool(args.visualize and not args.save_figs),
        "th_raw": args.threshold, # Pass the threshold to the pipeline
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
                    if args.save_figs:
                        # save the component topographies automatically
                        try:
                            figs = detection_result.ica.plot_components(picks=detection_result.accepted_components)
                            # If the MNE plotting returned a single figure or list, attempt to save
                            outdir = args.figs_outdir
                            os.makedirs(outdir, exist_ok=True)
                            if isinstance(figs, list):
                                for i, f in enumerate(figs):
                                    try:
                                        f.savefig(os.path.join(outdir, f"ica_topomap_{i}.png"), dpi=150)
                                        plt.close(f)
                                    except Exception:
                                        pass
                            else:
                                try:
                                    figs.savefig(os.path.join(outdir, "ica_topomaps.png"), dpi=150)
                                    plt.close(figs)
                                except Exception:
                                    pass
                        except Exception:
                            # fallback: just show and let user close
                            plt.show()
                    else:
                        plt.show()
            except Exception as e:
                print(f"Could not plot components: {e}")
            # (topomap handling done above)
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
        # Call the post-hoc diagnostic plotting helper (no need to rerun ICA)
        try:
            # Load manual spikes for visualization
            try:
                mat = loadmat(mat_file_path)
                manual_spikes = parse_spike_times(mat, label_marker="!")
            except Exception:
                manual_spikes = None

            comp_indices = None
            if args.plot_components:
                comp_indices = [int(x) for x in args.plot_components.split(',') if x.strip()]
            top_k = args.plot_topk
            # If saving figs, do not show GUI
            show_figs = not args.save_figs
            plot_detection_components(detection_result, component_indices=comp_indices, top_k=top_k,
                                      max_plot_seconds=30, save_figs=args.save_figs, outdir=args.figs_outdir, show_figs=show_figs,
                                      manual_spike_times=manual_spikes)
        except Exception as e:
            print(f"Diagnostic plotting failed: {e}")
    else:
        print("The pipeline did not return any results. Please check the logs for warnings or errors.")


if __name__ == "__main__":
    main()