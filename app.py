import streamlit as st_app  # renaming Streamlit to avoid conflict with spike_tools
import pandas as pd
import os
import matplotlib.pyplot as plt
from owl_model.analysis import data_loader as dl
from owl_model.analysis import spike_tools as st
import scipy.io as sio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
import numpy as np
from scipy.signal import find_peaks

# defining file directories
base_path = os.path.join(os.path.dirname(__file__), "merged_data")

paths = {
    "itd": os.path.join(base_path, "itd/"),
    "ild": os.path.join(base_path, "ild/"),
    "abl": os.path.join(base_path, "abl/"),
    "freq": os.path.join(base_path, "freq/"),
    "itdild": os.path.join(base_path, "itdiid/"),
}

# loading the dataset
@st_app.cache_data
def load_data():
    return pd.read_csv("complete_data.csv").fillna('')

df_files = load_data()

# Streamlit UI
st_app.title("Barn Owl Neuroscience Data Explorer")

# Navigation
page = st_app.sidebar.radio("Select Page", ["Home", "Visualize Data", "ITD-ILD Analysis", "Peak-Trough Ratio Analysis", "Peak-Trough Range Distribution"])

if page == "Home":
    st_app.subheader("Welcome to the ICCL response viewer!")

    # option to show entire dataset (optional)
    if st_app.checkbox("Show complete dataset"):
        st_app.dataframe(df_files)

elif page == "Visualize Data":
    st_app.subheader("Neuron Data Visualization")

    # sllow user to input Neuron ID
    neuron_index = st_app.number_input("Enter Neuron Index (0-114):", min_value=0, max_value=114, step=1)

    if neuron_index in df_files.index:
        st_app.write(f"Showing available plots for Neuron Index: {neuron_index}")
        st_app.dataframe(df_files.iloc[neuron_index])
    
        # extracting file paths dynamically
        plots_generated = False
        fig_list = []

        # ITD Plot
        itd_file_name = df_files.iloc[neuron_index]["itd_file_name"]
        if itd_file_name and isinstance(itd_file_name, str):
            itd_path = os.path.join(paths["itd"], itd_file_name)
            if os.path.exists(itd_path):
                ITD, mean_spike_count_itd, std_spike_count_itd = dl.load_itd_mat(itd_path)
                st.fit_itd_curve(ITD, mean_spike_count_itd, std_spike_count_itd, plot=True)
                fig_list.append(plt.gcf())
                plots_generated = True
                plt.close() 

        # ILD Plot
        ild_file_name = df_files.iloc[neuron_index]["ild_file_name"]
        if ild_file_name and isinstance(ild_file_name, str):
            ild_path = os.path.join(paths["ild"], ild_file_name)
            if os.path.exists(ild_path):
                ILD, mean_spike_count_ild, std_spike_count_ild = dl.load_ild_mat(ild_path)
                st.fit_ild_curve(ILD, mean_spike_count_ild, std_spike_count_ild, plot=True)
                fig_list.append(plt.gcf())
                plots_generated = True
                plt.close()

        # ABL Plot
        abl_file_name = df_files.iloc[neuron_index]["abl_file_name"]
        if abl_file_name and isinstance(abl_file_name, str):
            abl_path = os.path.join(paths["abl"], abl_file_name)
            if os.path.exists(abl_path):
                ABL, mean_spike_count_abl, std_spike_count_abl = dl.load_abl_mat(abl_path)
                st.fit_abl_curve(ABL, mean_spike_count_abl, std_spike_count_abl, plot=True)
                fig_list.append(plt.gcf())
                plots_generated = True
                plt.close()

        # Frequency Plot (BF)
        freq_file_name = df_files.iloc[neuron_index]["freq_file_name"]
        if freq_file_name and isinstance(freq_file_name, str):
            freq_path = os.path.join(paths["freq"], freq_file_name)
            if os.path.exists(freq_path):
                frequency, mean_spike_count_freq, std_spike_count_freq = dl.load_freq_mat(freq_path)
                st.fit_frequency_curve(frequency, mean_spike_count_freq, std_spike_count_freq, plot=True)
                fig_list.append(plt.gcf())
                plots_generated = True
                plt.close()

        # ITDILD Plot
        itdild_file_name = df_files.iloc[neuron_index]["itdild_file_name"]
        if itdild_file_name and isinstance(itdild_file_name, str):
            itdild_path = os.path.join(paths["itdild"], itdild_file_name)
            if os.path.exists(itdild_path):
                ITD, ILD, mean_spike_count_itdild, std_spike_count_itdild = dl.load_itdild_mat(itdild_path)

                fig = st.fit_itdild_matrix(ITD, ILD, mean_spike_count_itdild, plot=True)  
                st_app.plotly_chart(fig)
                plots_generated = True

        # displaying all generated plots
        if plots_generated:
            # st_app.subheader("Generated Plots")
            for fig in fig_list:
                st_app.pyplot(fig)
        else:
            st_app.warning("No valid data files found for this neuron.")

    else:
        st_app.warning("Invalid Neuron Index selected.")

# elif page == "Visualize Data":
#     st_app.subheader("Neuron Data Visualization")

#     # Allow user to input Neuron ID
#     neuron_index = st_app.number_input("Enter Neuron Index (0-114):", min_value=0, max_value=114, step=1)

#     if neuron_index in df_files.index:
#         st_app.write(f"Showing available plots for Neuron Index: {neuron_index}")
#         st_app.dataframe(df_files.iloc[[neuron_index]])

#         # Extract file paths dynamically
#         plots_generated = False
#         fig_list = []

#         # ITD Plot
#         itd_file_name = df_files.iloc[neuron_index]["itd_file_name"]
#         if itd_file_name and isinstance(itd_file_name, str):
#             itd_path = os.path.join(paths["itd"], itd_file_name)
#             if os.path.exists(itd_path):
#                 ITD, mean_spike_count_itd, std_spike_count_itd = dl.load_itd_mat(itd_path)
#                 st.fit_itd_curve(ITD, mean_spike_count_itd, std_spike_count_itd, plot=True)
#                 fig_list.append(plt.gcf())
#                 plots_generated = True

#         # ILD Plot
#         ild_file_name = df_files.iloc[neuron_index]["ild_file_name"]
#         if ild_file_name and isinstance(ild_file_name, str):
#             ild_path = os.path.join(paths["ild"], ild_file_name)
#             if os.path.exists(ild_path):
#                 ILD, mean_spike_count_ild, std_spike_count_ild = dl.load_ild_mat(ild_path)
#                 st.fit_ild_curve(ILD, mean_spike_count_ild, std_spike_count_ild, plot=True)
#                 fig_list.append(plt.gcf())
#                 plots_generated = True

#         # ABL Plot
#         abl_file_name = df_files.iloc[neuron_index]["abl_file_name"]
#         if abl_file_name and isinstance(abl_file_name, str):
#             abl_path = os.path.join(paths["abl"], abl_file_name)
#             if os.path.exists(abl_path):
#                 ABL, mean_spike_count_abl, std_spike_count_abl = dl.load_abl_mat(abl_path)
#                 st.fit_abl_curve(ABL, mean_spike_count_abl, std_spike_count_abl, plot=False)
#                 fig_list.append(plt.gcf())
#                 plots_generated = True

#         # Frequency Plot
#         freq_file_name = df_files.iloc[neuron_index]["freq_file_name"]
#         if freq_file_name and isinstance(freq_file_name, str):
#             freq_path = os.path.join(paths["freq"], freq_file_name)
#             if os.path.exists(freq_path):
#                 frequency, mean_spike_count_freq, std_spike_count_freq = dl.load_freq_mat(freq_path)
#                 st.fit_frequency_curve(frequency, mean_spike_count_freq, std_spike_count_freq, plot=False)
#                 fig_list.append(plt.gcf())
#                 plots_generated = True
    
#         # Display all generated plots
#         if plots_generated:
#             st_app.subheader("Generated Plots")
#             for fig in fig_list:
#                 st_app.pyplot(fig)
#         else:
#             st_app.warning("No valid data files found for this neuron.")

elif page == "ITD-ILD Analysis":
    st_app.subheader("ITD-ILD Analysis")

    # User input for running all neurons or selecting one manually
    # run_all_neurons = st_app.checkbox("Run All Neurons")

    neuron_index = st_app.number_input("Enter Neuron Index (0-75):", min_value=0, max_value=75, step=1)
    neuron_indices = [neuron_index]

    # ITD tolerance for filtering peaks
    itd_tolerance = st_app.slider("ITD Tolerance (µs)", min_value=10, max_value=200, value=50, step=5)

    # Spike count threshold slider
    spike_count_threshold = st_app.slider("Minimum Spike Count Threshold", min_value=0, max_value=5, value=2)

    # Initializing session state for storing slopes
    if "slopes" not in st_app.session_state:
        st_app.session_state.slopes = [] 

    for neuron_index in neuron_indices:
        if neuron_index in df_files.index:
            itdild_file_name = df_files.iloc[neuron_index]["itdild_file_name"]

            if itdild_file_name and isinstance(itdild_file_name, str):
                itdild_path = os.path.join(paths["itdild"], itdild_file_name)
                if os.path.exists(itdild_path):
                    
                    ITD, ILD, mean_spike_count_itdild, std_spike_count_itdild = dl.load_itdild_mat(itdild_path)

                    # Determine reference ITD from the curve with the largest peak-trough difference
                    peak_trough_diff = [
                        np.max(mean_spike_count_itdild[ild_idx]) - np.min(mean_spike_count_itdild[ild_idx]) 
                        for ild_idx in range(len(ILD))
                    ]

                    best_curve_index = np.argmax(peak_trough_diff)

                    # Computing reference ITD using Akima interpolation
                    reference_itd, _, _ = st.fit_itd_curve_3_peaks(
                        ITD, mean_spike_count_itdild[best_curve_index], std_spike_count_itdild[best_curve_index], plot=False
                    )

                    # Finding valid ILD-ITD pairs
                    valid_ild = []
                    valid_itd = []
                    details_output = []

                    # Select ILD Values to Include (Checkboxes)
                    selected_ilds = st_app.multiselect(
                        "Select ILD Values to Include:",
                        options=ILD,  # All ILD values available
                        default=ILD  # By default, all are selected
                    )

                    for ild_idx, ild in enumerate(ILD):
                        if ild not in selected_ilds:
                            continue  # Skip ILDs that are unchecked 
                        itd_curve = mean_spike_count_itdild[ild_idx]
                        std_dev = std_spike_count_itdild[ild_idx]

                        # Excluding plots if max spike count < spike_count_threshold
                        if np.max(itd_curve) < spike_count_threshold:
                            details_output.append(f"Neuron {neuron_index} | ILD={ild:.1f} dB | Max spike count below threshold | Rejected ✗")
                            continue

                        # Computing main, secondary, and third peaks using Akima interpolation
                        main_peak_itd, secondary_peak_itd, third_peak_itd = st.fit_itd_curve_3_peaks(
                            ITD, itd_curve, std_dev, plot=False
                        )

                        # Ensuring valid ITD values before proceeding
                        if main_peak_itd is None and secondary_peak_itd is None and third_peak_itd is None:
                            details_output.append(f"Neuron {neuron_index} | ILD={ild:.1f} dB | No valid peaks found | Rejected ✗")
                            continue  # Skip this ILD value

                        # Defaulting to main peak
                        chosen_peak_itd = main_peak_itd
                        peak_type = "Accepted ✓ | Main Peak"

                        # If secondary peak exists, checking which is closer to reference ITD
                        if secondary_peak_itd is not None and reference_itd is not None:
                            if abs(secondary_peak_itd - reference_itd) < abs(main_peak_itd - reference_itd):
                                chosen_peak_itd = secondary_peak_itd
                                peak_type = "Accepted ✓ | Secondary Peak"

                        # If third peak exists, checking which is closer to reference ITD
                        if third_peak_itd is not None and reference_itd is not None:
                            if abs(third_peak_itd - reference_itd) < abs(chosen_peak_itd - reference_itd):
                                chosen_peak_itd = third_peak_itd
                                peak_type = "Accepted ✓ | Third Peak"

                        # Ensuring chosen_peak_itd is valid before using it
                        if chosen_peak_itd is not None and reference_itd is not None:
                            if abs(chosen_peak_itd - reference_itd) <= itd_tolerance:
                                valid_ild.append(ild)
                                valid_itd.append(chosen_peak_itd)
                            else:
                                peak_type = "Rejected ✗ (Far from reference ITD)"
                        else:
                            peak_type = "Rejected ✗ (No valid ITD)"

                        details_output.append(
                            f"ILD={ild:.1f} dB | Best ITD={chosen_peak_itd:.2f} µs | {peak_type}"
                            if chosen_peak_itd is not None
                            else f"ILD={ild:.1f} dB | Best ITD=N/A µs | {peak_type}"
                        )

                    # Regression Plot (only if running a single neuron)
                    if not run_all_neurons and len(valid_ild) > 1:
                        slope, intercept, r_value, p_value, std_err = linregress(valid_ild, valid_itd)

                        # Store slopes dynamically
                        st_app.session_state.slopes.append(slope)
                        st_app.text(slope)

                        regression_line = slope * np.array(valid_ild) + intercept
                        plt.figure(figsize=(9, 6))
                        plt.scatter(valid_ild, valid_itd, color='purple', edgecolors='black', s=80, label='Valid ITD-ILD Pairs', alpha=0.8)
                        plt.plot(valid_ild, regression_line, color='darkorange', linestyle='-', linewidth=2, 
                                 label=f'Regression Line\nSlope: {slope:.2f} µs/dB')
                        plt.xlabel('ILD (dB)', fontsize=14, fontweight='bold')
                        plt.ylabel('Best ITD (µs)', fontsize=14, fontweight='bold')
                        plt.title(f'Neuron {neuron_index}: ITD-ILD Regression', fontsize=16)
                        plt.xticks(fontsize=12)
                        plt.yticks(fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend(fontsize=12, frameon=True, loc='best')
                        st_app.pyplot(plt)
                        plt.close()
                    elif run_all_neurons and len(valid_ild) > 1:
                        # Store slopes for all neurons
                        slope, _, _, _, _ = linregress(valid_ild, valid_itd)
                        st_app.session_state.slopes.append(slope)

                else:
                    st_app.warning(f"No ITD-ILD data available for neuron {neuron_index}.")

 # See More Details" Section- only if running a single neuron)
    if not run_all_neurons and st_app.button("See More Details"):
        st_app.text(f"Reference ITD: {reference_itd:.1f} µs")
        col1, col2, col3, col4 = st_app.columns(4)

        for i, ild in enumerate(ILD):
            itd_curve = mean_spike_count_itdild[i]
            std_dev = std_spike_count_itdild[i]

            # Computing main, secondary, and third peaks for visualization
            main_peak_itd, secondary_peak_itd, third_peak_itd = st.fit_itd_curve_3_peaks(
                ITD, itd_curve, std_dev, plot=True
            )

            # Showing ILD value in title
            plt.title(f"ILD = {ild:.1f} dB", fontsize=10)
            fig = plt.gcf()

            # Dynamically placing plots into columns
            if i % 4 == 0:
                with col1:
                    st_app.pyplot(fig)
            elif i % 4 == 1:
                with col2:
                    st_app.pyplot(fig)
            elif i % 4 == 2:
                with col3:
                    st_app.pyplot(fig)
            else:
                with col4:
                    st_app.pyplot(fig)

            plt.close()

        st_app.text("\n".join(details_output))

elif page == "Peak-Trough Ratio Analysis":
    st_app.subheader("Peak-Trough Ratio Analysis")

    neuron_index = st_app.number_input("Enter Neuron Index (0-75):", min_value=0, max_value=75, step=1)

    if neuron_index in df_files.index:
        itdild_file_name = df_files.iloc[neuron_index]["itdild_file_name"]

        if itdild_file_name and isinstance(itdild_file_name, str):
            itdild_path = os.path.join(paths["itdild"], itdild_file_name)
            if os.path.exists(itdild_path):

                ITD, ILD, mean_spike_count_itdild, std_spike_count_itdild = dl.load_itdild_mat(itdild_path)

                raw_peak_trough_diffs = []
                peak_trough_ratios = []
                valid_ilds = []

                # Spike threshold to exclude bad curves
                spike_threshold = st_app.slider("Minimum Peak Spike Count (Spikes)", 1, 10, 2)

                for ild_idx, ild in enumerate(ILD):
                    itd_curve = mean_spike_count_itdild[ild_idx]

                    peak = np.max(itd_curve)
                    trough = np.min(itd_curve)

                    if peak >= spike_threshold:
                        raw_peak_trough_diffs.append(peak - trough)
                        peak_trough_ratios.append((peak - trough) / peak)
                        valid_ilds.append(ild)

                if len(valid_ilds) > 0:
                    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

                    # Left plot: raw peak-trough difference
                    axs[0].plot(valid_ilds, raw_peak_trough_diffs, marker='o', linestyle='-', color='orange')
                    axs[0].set_xlabel('ILD (dB)', fontsize=12)
                    axs[0].set_ylabel('Peak - Trough (spikes)', fontsize=12)
                    axs[0].set_title('Raw Peak-Trough Difference', fontsize=14)
                    axs[0].grid(True, linestyle='--', alpha=0.5)

                    # Right plot: normalized peak-trough ratio
                    axs[1].plot(valid_ilds, peak_trough_ratios, marker='o', linestyle='-', color='blue')
                    axs[1].set_xlabel('ILD (dB)', fontsize=12)
                    axs[1].set_ylabel('(Peak - Trough) / Peak', fontsize=12)
                    axs[1].set_title('Normalized Peak-Trough Ratio', fontsize=14)
                    axs[1].grid(True, linestyle='--', alpha=0.5)

                    plt.tight_layout()
                    st_app.pyplot(fig)
                    # Also show the full ITD-ILD matrix plot below
                    fig_matrix = st.fit_itdild_matrix(ITD, ILD, mean_spike_count_itdild, plot=True)
                    st_app.subheader("ITD-ILD Matrix Plot")
                    st_app.plotly_chart(fig_matrix)

                    plt.close()

                else:
                    st_app.warning("No valid ITD curves after applying spike threshold.")
            else:
                st_app.warning("ITD-ILD file does not exist for this neuron.")
        else:
            st_app.warning("No ITD-ILD data available for this neuron.")

elif page == "Peak-Trough Range Distribution":
    st_app.subheader("Distribution of Peak-Trough Ranges Across Neurons")

    spike_threshold = st_app.slider("Minimum Peak Spike Count (Spikes)", 1, 10, 2)

    range_values = []

    for neuron_index in df_files.index:
        itdild_file_name = df_files.loc[neuron_index, "itdild_file_name"]
        if itdild_file_name and isinstance(itdild_file_name, str):
            itdild_path = os.path.join(paths["itdild"], itdild_file_name)
            if os.path.exists(itdild_path):
                ITD, ILD, mean_spike_count_itdild, std_spike_count_itdild = dl.load_itdild_mat(itdild_path)

                ratios = []
                for itd_curve in mean_spike_count_itdild:
                    peak = np.max(itd_curve)
                    trough = np.min(itd_curve)
                    if peak >= spike_threshold:
                        ratio = (peak - trough) / peak
                        ratios.append(ratio)

                if len(ratios) > 0:
                    range_val = max(ratios) - min(ratios)
                    range_values.append(range_val)

    if len(range_values) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(range_values, bins=15, color='mediumseagreen', edgecolor='black', alpha=0.8)
        plt.xlabel("Range of (Peak - Trough) / Peak", fontsize=12)
        plt.ylabel("Number of Neurons", fontsize=12)
        plt.title("Histogram of Peak-Trough Ratio Ranges", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st_app.pyplot(plt)
        plt.close()
    else:
        st_app.warning("No valid data found for the current threshold.")

else:
    st_app.warning("No ITD-ILD data available for this neuron.")