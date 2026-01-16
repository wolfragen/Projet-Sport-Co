# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 20:08:26 2025

@author: konra
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_csv(main_path, csv_name):
    """
    Loads a CSV containing training data arrays and creates
    one plot per column using matplotlib.pyplot.
    """

    csv_path = main_path + csv_name
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.replace({None: pd.NA})
    
    cols = df.columns
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))  # 3x3 grid
    axes = axes.flatten()  # easier indexing
    
    for i, column in enumerate(cols):
        ax = axes[i]
        ax.plot(df[column])
        ax.set_title(column)
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
    
    plt.tight_layout()
    plt.savefig(main_path + "training_plot.png")
    plt.close()
    return

plot_training_csv("C:/.ingé/Projet-Sport-Co-Networks/fail=0.02/", "training_data.csv")



















