import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Monte Carlo Approximation of Pi

sns.set(style='whitegrid', context="talk")

def estimate_pi(num_points: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    
    inside_circle = x**2 + y**2 <= 1
    
    pi_estimate = 4 * np.mean(inside_circle)
    
    df = pd.DataFrame({
        "x" : x,
        "y" : y,
        "inside_circle" : inside_circle
    })
    
    return pi_estimate, df

def plot_points(df, pi_estimate, num_points):
    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data = df,
        x = "x",
        y = "y",
        hue = "inside_circle",
        palette = {True: "seagreen", False: "lightcoral"},
        s = 10,
        alpha = 0.7
    )
    
    theta = np.linspace(0, np.pi/2, 300)
    plt.plot(np.cos(theta), np.sin(theta), color = 'black', linewidth = 2)
    
    plt.title(f"Monte Carlo Pi Approximation (N={num_points})\nEstimated Pi = {pi_estimate:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend(title="Inside Circle", loc="upper right")
    plt.show()
    
def run_multiple_trials(trials = 30, points_per_trial = 100_000):
    results = []
    for i in range(trials):
        pi_est, _ = estimate_pi(points_per_trial)
        results.append(pi_est)
        
    df_results = pd.DataFrame({"Trial": range(1, trials + 1), "Pi_Estimate": results})
    
    plt.figure(figsize = (8, 5))
    sns.histplot(df_results["Pi_Estimate"], bins = 10, kde = True, color = "skyblue")
    plt.axvline(np.pi, color  = "red", linestyle = "--", label = "Actual Pi")
    plt.title(f"Distribution of Pi Estimates ({trials} trials, {points_per_trial} points each)")
    plt.xlabel("Estimated Pi")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    return df_results

N = 100_000

pi_estimate, df = estimate_pi(N, seed  = 69)
print(f"Estimated Pi = {pi_estimate:.6f}")

plot_points(df, pi_estimate, N)
results_df = run_multiple_trials(trials = 30, points_per_trial = N)
print(results_df.describe())
    