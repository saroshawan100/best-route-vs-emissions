# Best Route VS. Best Emissions

An AI-driven routing system designed to compare standard time-optimized navigation against a lowest-emissions alternative using Spatio-Temporal Adaptive Embedding transformers (STAEformer) and SUMO simulation.

## Architecture
* **Forecasting:** BasicTS / STAEformer (LargeST Dataset)
* **Routing:** A* search via OpenStreetMap
* **Simulation:** Eclipse SUMO (Actuated Signals & HBEFA Emissions)
* **Optimization:** Green Light Optimal Speed Advisory (GLOSA)
