# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 02:08:19 2024

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the data
file_path = 'data4.csv'  # Adjust the path if needed
data = pd.read_csv(file_path, header=None)
data.columns = ['Salary']

# Calculate the KDE for the salary data
data_array = data['Salary'].values
kde = gaussian_kde(data_array)

# Estimating the PDF values for each data point
pdf_values = kde.evaluate(data_array)

# Calculate the PDF-weighted mean salary
weighted_mean_salary = np.sum(data_array * pdf_values) / np.sum(pdf_values)

# Calculate the value X such that 25% of people have a salary below X
X = np.percentile(data_array, 25)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(data_array, bins='auto', density=True, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(weighted_mean_salary, color='red', linestyle='dashed', linewidth=2, label=f'PDF-weighted Mean Salary: {weighted_mean_salary:.2f}')
plt.axvline(X, color='green', linestyle='dashed', linewidth=2, label=f'X (Bottom 25% Salary): {X:.2f}')

# Adding details to the plot
plt.title('Probability Density Function of Annual Salaries')
plt.xlabel('Annual Salary')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
