# Tesla_Vehicle_Noise_Anomaly_Detection_System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org/) [![C++](https://img.shields.io/badge/C%2B%2B-17%2B-green.svg)](https://isocpp.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a comprehensive Machine Learning and high-performance computing solution for detecting noise, vibration, and harshness (NVH) anomalies in Tesla vehicles. It includes a custom CNN model for audio classification, synthetic data generation, interactive dashboards, and a production-grade C++ inference engine with SIMD optimizations. demonstrates end-to-end ML engineering from data pipelines to real-time deployment.

## Features
- **Deep Learning Model**: Custom CNN with attention mechanisms for sound event detection/classification, trained on synthetic Tesla audio data (e.g., brake squeaks, motor rattles).
- **Audio Processing**: Mel-spectrogram generation using Librosa, with anomaly simulation for Tesla models (Model 3, S, X, Cybertruck).
- **Interactive Dashboard**: Streamlit-based UI for real-time fleet monitoring, visualizations (Plotly heatmaps, maps), and anomaly alerts.
- **C++ Inference Engine**: Multi-threaded, lock-free pipeline with AVX2 SIMD for spectrogram computation and OpenGL for real-time visualization, achieving <10ms inference.
- **Fleet Simulation & Analytics**: Multi-threaded Python simulator for 1,500+ virtual vehicles, with performance benchmarking and SHAP interpretability.
- **Production Readiness**: TorchScript model export, JSON configs, and benchmarks for scalability across millions of Tesla vehicles.

## Project Structure
- `Tesla_NVH_Audio_Detection.ipynb`: Jupyter notebook for model training, data generation, and interactive demos.
- `tesla_dashboard.py`: Streamlit dashboard for real-time Tesla fleet diagnostics.
- `tesla_final_demo.py`: End-to-end ML demo with inference and performance reporting.
- `tesla_fleet_manager.py`: Multi-threaded fleet simulator for enterprise-scale processing.
- `tesla_model_explorer.py`: Model analysis tool with interpretability (SHAP) and optimization recommendations.
- `tesla_nvh_advanced.cpp`: C++ inference engine with SIMD, threading, and visualization.
- `tesla_nvh_model.pt`: Exported TorchScript model (simulated path; generate via notebook).
- `requirements.txt`: Python dependencies.


## Installation:
1. Clone the repository:
- git clone https://github.com/Alibek-07/Tesla_Vehicle_Noise_Anomaly_Detection_System.git
- cd Tesla_NVH_Audio_Detection.ipynb
- cd tesla_dashboard.py
2. Install Python dependencies:
- pip install -r requirements.txt
3. For C++ components (requires GCC/Clang with AVX2 support):
- Compile `tesla_nvh_advanced.cpp` (e.g., `g++ -O3 -mavx2 -std=c++17 tesla_nvh_advanced.cpp -o tesla_nvh -lOpenGL -lGLUT`).
- Ensure LibTorch is installed for C++ PyTorch integration (download from [PyTorch website](https://pytorch.org/)).

## Usage
### Python Components
- Run the Jupyter notebook: Tesla_NVH_Audio_Detection.ipynb
- Train the model, generate data, and run interactive demos.
- Launch the Streamlit dashboard: run tesla_dashboard.py
- Run fleet simulation: python tesla_fleet_manager.py
- Model analysis: python tesla_model_explorer.py

### C++ Inference
- Run the compiled executable: ./tesla_nvh tesla_nvh_model.pt --benchmark --visualize
- - Options: `--benchmark` for performance tests, `--visualize` for OpenGL spectrograms, `--performance` for metrics.

## Results & Benchmarks
- Model Accuracy: 87.6% on simulated data.
- Inference Time: <10ms (C++ SIMD-optimized).
- Fleet Scale: Handles 1,500+ vehicles with multi-threading.
- See `tesla_demo_results.json` and `tesla_fleet_analysis.json` for detailed reports.

## Contributing
Contributions are welcome! Please fork and submit a pull request. Focus on improving ML accuracy, C++ optimizations, or Tesla-specific features.

## License
This project is licensed under the MIT License 

## Acknowledgments
- Built for Tesla NVH Internship submission (January 2026).
- Inspired by real-world automotive diagnostics challenges.
- Uses open-source libraries like PyTorch, Librosa, and Streamlit.

For questions, contact [alibekd0725@gmail.com].



