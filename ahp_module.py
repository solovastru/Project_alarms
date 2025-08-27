import numpy as np


def ahp_weights(dictionary): 
    labels = list({key[0] for key in dictionary.keys()})
    matrix = np.array([[dictionary[(i,j)] for j in labels] for i in labels], dtype=float)

    # Normalize columns
    col_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / col_sums

    # Compute priority vector (row averages)
    priority_vector = normalized_matrix.mean(axis=1)

    # put the weights into a dictionary
    weights_dict = {labels[i]: round(priority_vector[i], 3) for i in range(len(labels))}
    weights_dict = dict(sorted(weights_dict.items()))

    # Show weights

    return weights_dict



def consistency_ratio_dict(weights_dict, comparison_dict):
    """
    Calculate CI and CR for an AHP matrix using dictionaries.

    Parameters:
    - priority_dict: dict with keys = alternatives, values = priority weights
    - comparison_dict: dict of pairwise comparisons, keys = (i,j) tuples, values = ratios
    """
    labels = list(weights_dict.keys())
    n = len(labels)

    # Build matrix from dictionary
    matrix = np.array([[comparison_dict[(i,j)] for j in labels] for i in labels], dtype=float)

    # Convert priority dict to vector
    priority_vector = np.array([weights_dict[label] for label in labels])

    # Weighted sum vector
    weighted_sum = matrix.dot(priority_vector)

    # Lambda max
    lambda_max = (weighted_sum / priority_vector).mean()
    print(lambda_max)
    # Consistency Index
    CI = round((lambda_max - n) / (n - 1), 3)

    # Random Index (Saaty)
    RI_dict = {1:0,2:0,3:0.58,4:0.9,5:1.12,6:1.24,7:1.32,
               8:1.14,9:1.45,10:1.49,11:1.51,12:1.48,13:1.56,
               14:1.57,15:1.59,16:1.605,17:1.61,18:1.615,19:1.62,20:1.625}

    CR = round(CI / RI_dict[n], 2)

    print(f'Consistency Index (CI): {CI}')
    print(f'Consistency Ratio (CR): {CR}')

    if CR < 0.1:
        print("The matrix is consistent ✅")
    else:
        print("The matrix is NOT consistent ❌")


