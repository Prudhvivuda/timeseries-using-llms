# Time Series Forecasting with LLMs

## Overview  
This repository demonstrates how to use large language models (LLMs) to perform time series forecasting. We benchmark LLM‑based approaches against more traditional methods and explore techniques to improve accuracy, efficiency, and long‑horizon stability.

## Key Features  
- Benchmarking of LLMs (e.g. Mistral‑7B) vs classical models (e.g. Prophet, TimesFM)  
- Sliding‑window and multivariate input pipelines for better temporal context  
- Experimentation with diffusion‑based denoisers and residual finetuning to enhance predictions  
- Evaluation metrics include RMSE, MAE, and R²  

## Getting Started

### Requirements  
- Python 3.8+  
- Jupyter / JupyterLab  
- Key libraries:  
  ```bash
  pip install torch transformers diffusers prophet pandas numpy matplotlib
  ```
- (Optional) GPU for faster training / inference  

### Files  
- `batch_process.ipynb`: data loading, preprocessing, sliding‑window forecasts  
- `diffusion_model.ipynb`: models using diffusion‑based denoisers  
- `finetuning.ipynb`: fine‑tuning LLMs for time series tasks  
- `residual_finetuning.ipynb`: residual corrections on baseline forecasts  
