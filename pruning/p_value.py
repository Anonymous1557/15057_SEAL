import argparse
import math

import numpy as np

from scipy.special import gammaln
from tqdm import tqdm

def log_binom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def log_prob(n, k, p):
    return log_binom(n, k) + k * np.log(p) + (n - k) * np.log(1 - p)

def log_p_value(n, k, p):
    log_p_value = -np.inf
    for i in range(k, n + 1):
        log_p = log_prob(n, i, p)
        log_p_value = np.logaddexp(log_p_value, log_p)
    return log_p_value

def main():
    parser = argparse.ArgumentParser(description='Calculate the log probability for a given BER and n.')
    parser.add_argument('n', type=int, help='Number of trials (n)')
    parser.add_argument('ber', type=float, help='Bit Error Rate (BER)')
    args = parser.parse_args()

    n = args.n
    p = 0.5
    if args.ber > 100.:
        raise ValueError("BER must be between 0 and 100.")
    ber = args.ber / 100.
    k = int((1 - ber) * n)

    log_p_val = log_p_value(n, k, p)

    log10_val = log_p_val / np.log(10)
    fractional, integer = math.modf(log10_val)
    
    if fractional < 0:
        fractional += 1
        integer -= 1

    mantissa = 10 ** fractional
    exponent = int(integer)


    print(f"log(P(X >= {k})) = {log_p_val}")
    print(f"P(X >= {k}) ≈ 10^{log10_val:.2f} ≈ {mantissa:.2f} x 10^{exponent}")

if __name__ == "__main__":
    main()
