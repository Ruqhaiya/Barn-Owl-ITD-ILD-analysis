# 🦉 Barn Owl Neuroscience Data Explorer 

## Overview

This Streamlit app is a scientific data visualization and analysis tool built to explore neural response data from the **inferior colliculus central nucleus (ICcl)** of the barn owl's auditory midbrain. The ICcl plays a key role in spatial hearing, integrating cues like interaural time difference (ITD), interaural level difference (ILD), amplitude (ABL), and frequency tuning. This app provides an interactive tool for researchers to inspect neural tuning across these dimensions.

The project stems from ongoing research aiming to model the diversity of auditory response characteristics in barn owls. This tool supports hypothesis generation, data exploration, and summarization of neuron-level tuning curves and regression properties. 

> 🔗 **[Launch the App](https://iccl-response-viewer.streamlit.app/)**  

## This project demonstrates end-to-end skills in:

* Neuroscience data processing
* Interactive web app development (Streamlit, Plotly, Matplotlib)
* Data analysis and visualization
* Curve fitting and signal characterization
* Communication of scientific results
* Collaboration on neuroscience research

It serves as a showcase of both scientific understanding and software engineering.

## Application Structure

### 1. Home

* Provides an introduction to the app.
* Option to view the full dataset of available neuron recordings, this dataset is basically metadata of filepaths to those recordings. Meaning, this is what we use to locate a file for a particular neuron and the files are stored in a directory called 'merged data'. 

![image](https://github.com/user-attachments/assets/1cee0193-4c1f-464a-8dcf-618416c2c0f0)

### 2. Visualize Data

* **Input**: User selects a neuron index (0-114).
* **Output**: Displays all available tuning curves for that neuron, including:

  * **ITD**: Interaural Time Difference tuning curve.
  * **ILD**: Interaural Level Difference tuning curve.
  * **ABL**: Amplitude Level tuning.
  * **Frequency**: Frequency tuning curve.
  * **ITD-ILD Matrix**: 2D heatmap showing ITD response at various ILD levels.
![Data Visualization Page Demo] (assets/viz-data.gif)

### 3. ITD-ILD Analysis

* **Purpose**: Perform slope-based analysis of ITD peak locations across ILDs.
* **Features**:

  * Akima interpolation is used to detect up to 3 peaks per ITD curve.
  * Reference ITD is determined using the strongest response curve (largest peak-trough difference).
  * Filters valid peaks based on closeness to reference ITD.
  * Fits a linear regression to valid (ILD, ITD) pairs.

### 4. Peak-Trough Ratio Analysis

* **Purpose**: Analyze the magnitude and shape of each ITD curve.
* **Features**:

  * Computes raw peak-to-trough difference and normalized peak-trough ratio for each ILD.
  * Filters curves below a spike count threshold.
  * Displays:

    * Line plot of raw differences.
    * Line plot of normalized ratios.
    * Corresponding ITD-ILD heatmap.

### 5. Peak-Trough Range Distribution

* **Purpose**: Explore distribution of response curve dynamic range across neurons.
* **Features**:

  * For each neuron, computes the range of normalized peak-trough ratios.
  * Plots a histogram of these range values.
  * Useful for identifying neurons with highly tuned vs. flat responses.

## Technologies Used

* **Python 3.12**
* **Streamlit** for web UI
* **Matplotlib / Plotly** for plotting
* **Pandas / NumPy / SciPy** for data wrangling

## Getting Started

1. Clone the repo:

```bash
git clone https://github.com/yourusername/barn-owl-itd-ild-analysis.git
cd barn-owl-itd-ild-analysis
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```
---

For further details or to collaborate, feel free to reach out or explore the project. 
