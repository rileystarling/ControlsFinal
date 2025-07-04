# ControlsFinal

README - Pedestrian Traffic Project
===========================

Author: Riley Starling

UNI: rs4635

Date: 14 May 2025

Project Overview
===========================

This project implements an adaptive traffic control system using Model Predictive Control (MPC) on simulated intersections using UXsim. The goal is to reduce congestion and improve flow by dynamically adjusting signal phases based on real-time queue information.

The simulation analyzes queue lengths, vehicle wait times, and MPC computation time across various intersection types.

How to Run the Code and Generate Figures
===========================

1. Requirements:
   - Python 3.x
   - itertools, matplotlib, collections, time, random
   - UXsim (Open-Source, easy to download)
              
   
2. Run In terminal:
   conda activate {replace_with_your_uxsim_env_name}
   python main.py
   
   This will:
    - Simulate the data
    - Run MPC
    - Generate and save relevant plots in the `figures/` folder
    - All three simulations run 15 iterations each
    - Don't worry if it takes a long time !
    
3. Alternative
   run MPC_simulator_final_standard.ipynnb, MPC_simulator_final_oneway.ipynb, MPC_simulator_final_T.ipynb

References
===========================

- UXsim GitHub: https://github.com/toruseo/UXsim.git
