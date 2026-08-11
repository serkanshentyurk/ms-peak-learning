import numpy as np
from scipy.stats import norm


def r_from_z(z):
    r = (np.exp(2*z)-1) / (np.exp(2*z)+1)
    return r 
def r_to_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def find_threshold_z(obj, r, significance_level=0.05):
    z1 = r_to_z(r)
    n = obj.data.shape[1] * obj.data.shape[2]
    # Calculate standard error
    standard_error = 1 / (n ** 0.5)
    
    # Find the critical z-value for the desired significance level
    critical_z_value = norm.ppf(1 - significance_level)
    
    # Calculate the minimum z2 value
    min_z2 = z1 + critical_z_value * standard_error
    min_r2 = r_from_z(min_z2)
    return min_r2

def fisher_z_test(r1, n1, r2, n2 = None, verbose = True):
    if n2 == None:
        n2 = n1
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_diff = (z1 - z2) / se_diff

    p_value = norm.cdf(z_diff)
    
    if verbose:
        print(f"Z-Difference: {z_diff}")
        print(f"P-Value: {p_value}")

        if p_value < 0.05:
            print(f"\nThe correlation r2 is significantly smaller than r1:\t p-value = {round(p_value,4)}")
        else:
            print("The difference is not statistically significant.")

    return z_diff, p_value