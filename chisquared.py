#Here we understand if there is a significative difference of the 5 most common POS of the Antisocial DATA

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Data provided
categories = ['Nouns', 'Conjunctions', 'Determiners', 'Personal pronouns', 'Adjectives']
n = [5334, 3716, 2517, 2506, 2210]
f = [6711, 3762, 2870, 1987, 2612]

# Lists to store chi-square statistics and p-values
chi2_values = []
p_values = []

# Total sample sizes
total1 = 24273
total2 = 23659

# Combine categories, n, f into a list of tuples
data = list(zip(categories, n, f))

# Perform chi-square test for each category
for category, n_val, f_val in data:
    observed = np.array([[n_val, total1 - n_val], [f_val, total2 - f_val]])
    
    # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(observed)
    
    # Append chi-square statistic and p-value to lists
    chi2_values.append(chi2)
    p_values.append(p)

    # Output results (optional)
    print(f"For category {category}:")
    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p}")
    print()

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(range(len(categories)), p_values, marker='o', linestyle='-', color='b', label='P-value')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (alpha = 0.05)')
plt.xlabel('Category')
plt.ylabel('P-value')
plt.title('P-value vs. Category')

# Set x-axis ticks and labels
plt.xticks(range(len(categories)), categories)

# Set y-axis limits to focus on the range 0 to 0.1
plt.ylim(0, 0.1)

plt.legend()
plt.grid(True)
plt.tight_layout()

# Annotate points with their respective values of n and f
for i, (category, p_val) in enumerate(zip(categories, p_values)):
    plt.annotate(f'n={n[i]}, f={f[i]}', (i, p_val), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
