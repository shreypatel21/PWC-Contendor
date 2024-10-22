import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define sigmoid and exponential functions
def sigmoid_func(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def exponential_func_new(x, A, B, C):
    return A + B * np.exp(C * x)

# Data
x_values = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
iron_ore_feed = np.array([8.7, 8.5, 8.3, 8.1, 7.8, 7.5, 7.2, 6.8])
dolomite_consumption = np.array([90, 88, 86, 84, 82, 80, 78, 76])
campaign_days = np.array([62.5, 62.5, 57.5, 52.5, 47.5, 47.5, 42.5, 41])
shutdown_days = np.array([11, 11, 11, 9, 8, 7, 7, 7])
dolochar_gen = np.array([0.25, 0.25, 0.27, 0.29, 0.31, 0.33, 0.34, 0.35])
steam_generation = np.array([6.5, 6.8, 7.2, 7.4, 7.6, 7.9, 8.1, 8.2])
overhead_cost = np.array([1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200])
dolochar_rate = np.array([500, 500, 500, 500, 450, 400, 350, 350])

# Curve fitting equations
p_iron_ore = np.polyfit(x_values, iron_ore_feed, 2)
iron_ore_fit = np.poly1d(p_iron_ore)

p_dolomite = np.polyfit(x_values, dolomite_consumption, 1)
dolomite_fit = np.poly1d(p_dolomite)

p_campaign_days = np.polyfit(x_values, campaign_days, 3)
campaign_days_fit = np.poly1d(p_campaign_days)

params_sigmoid_shutdown, _ = curve_fit(sigmoid_func, x_values, shutdown_days, p0=[max(shutdown_days), np.median(x_values), 1, min(shutdown_days)])

p_dolochar = np.polyfit(x_values, dolochar_gen, 3)
dolochar_fit = np.poly1d(p_dolochar)

# Steam generation using exponential function with new parameters
A_steam, B_steam, C_steam = 9.54890, -0.9215, 1.19951

p_overhead = np.polyfit(x_values, overhead_cost, 1)
overhead_fit = np.poly1d(p_overhead)

params_sigmoid_dolochar, _ = curve_fit(sigmoid_func, x_values, dolochar_rate, p0=[max(dolochar_rate), np.median(x_values), 1, min(dolochar_rate)])

# Constants based on image data
selling_price_AB_DRI = 30000  # INR/MT for A/B Grade DRI
selling_price_minus1mm_DRI = 18500  # INR/MT for -1 mm DRI
dolomite_cost_per_kg = 1.5  # INR/Kg

# Function to compute total profit
def compute_profit(x):
    # Compute the operational parameters using curve fits
    iron_ore_feed_rate = iron_ore_fit(x)
    dolomite_consumption = dolomite_fit(x)
    campaign_days = campaign_days_fit(x)
    shutdown_days = sigmoid_func(x, *params_sigmoid_shutdown)
    dolochar_generation = dolochar_fit(x)
    steam_generation = exponential_func_new(x, A_steam, B_steam, C_steam)
    overhead_cost = overhead_fit(x)
    dolochar_rate = sigmoid_func(x, *params_sigmoid_dolochar)
    
    # Available days
    available_days = 365 - (365 / campaign_days) * shutdown_days
    
    # Production per day (MT)
    production_per_day = iron_ore_feed_rate * 24 * 0.555
    
    # Annual production (MT)
    annual_production = production_per_day * available_days
    
    # Revenue from DRI (95% is A/B Grade DRI, 5% is -1 mm DRI)
    revenue_AB_DRI = (annual_production * 0.95) * selling_price_AB_DRI
    revenue_minus1mm_DRI = (annual_production * 0.05) * selling_price_minus1mm_DRI
    total_revenue_dri = revenue_AB_DRI + revenue_minus1mm_DRI
    
    # Dolochar generation and revenue
    total_dolochar = annual_production * dolochar_generation
    revenue_dolochar = total_dolochar * dolochar_rate
    
    # Steam generation and revenue
    total_steam = steam_generation * 24 * available_days
    revenue_steam = total_steam * 500
    
    # Cost calculations
    iron_ore_consumed = iron_ore_feed_rate * 24 * available_days
    cost_iron_ore = iron_ore_consumed * 8400
    
    # Coal cost
    avg_cfe_ratio = x * 0.402 + (1 - x) * 0.430
    avg_fc_coal = x * 0.52 + (1 - x) * 0.42
    avg_moisture_coal = x * 0.09 + (1 - x) * 0.08
    mc_coal = 1 - avg_moisture_coal
    average_feed_rate_coal = (iron_ore_feed_rate * avg_cfe_ratio * 0.64 * 0.97) / (avg_fc_coal * mc_coal)
    coal_consumed = average_feed_rate_coal * 24 * available_days
    cost_coal = coal_consumed * (x * 9800 + (1 - x) * 5000)
    
    # Dolomite cost (updated to reflect 1.5 INR/Kg from image)
    total_dolomite_consumed = (dolomite_consumption / 1000) * annual_production
    cost_dolomite = total_dolomite_consumed * dolomite_cost_per_kg * 1000  # Converting to INR
    
    # Overhead cost
    total_overhead_cost = overhead_cost * available_days
    
    # Total profit calculation
    profit = (total_revenue_dri + revenue_dolochar + revenue_steam) - (cost_iron_ore + cost_coal + cost_dolomite + total_overhead_cost)
    return profit

# Run the profit computation for a given value of x (fraction of imported coal)
x_values_to_evaluate = np.linspace(0.3, 1.0, 1000)
profits = np.array([compute_profit(x) for x in x_values_to_evaluate])

# Find the maximum profit and corresponding x
max_profit = np.max(profits)
optimal_x = x_values_to_evaluate[np.argmax(profits)]

print(f"Optimal Fraction of Imported Coal: {optimal_x:.4f}")
print(f"Maximum Profit: {max_profit:.2f} INR/year")

# Plot the parameter fits and profit function
fig, axs = plt.subplots(4, 2, figsize=(14, 16))

# Iron Ore Feed Rate plot
axs[0, 0].scatter(x_values, iron_ore_feed, color='blue', label='Data')
axs[0, 0].plot(x_values_to_evaluate, iron_ore_fit(x_values_to_evaluate), color='red', label='Quadratic Fit')
axs[0, 0].set_title('Iron Ore Feed Rate (Quadratic)')
axs[0, 0].set_xlabel('Fraction of Imported Coal')
axs[0, 0].set_ylabel('Feed Rate (MT/hr)')
axs[0, 0].legend()
axs[0, 0].text(0.5, 7.5, f"y = {p_iron_ore[0]:.2f}x^2 + {p_iron_ore[1]:.2f}x + {p_iron_ore[2]:.2f}")

# Dolomite Consumption plot
axs[0, 1].scatter(x_values, dolomite_consumption, color='blue', label='Data')
axs[0, 1].plot(x_values_to_evaluate, dolomite_fit(x_values_to_evaluate), color='red', label='Linear Fit')
axs[0, 1].set_title('Dolomite Consumption (Linear)')
axs[0, 1].set_xlabel('Fraction of Imported Coal')
axs[0, 1].set_ylabel('Dolomite Consumption (Kg/MT)')
axs[0, 1].legend()
axs[0, 1].text(0.5, 82, f"y = {p_dolomite[0]:.2f}x + {p_dolomite[1]:.2f}")

# Campaign Days plot
axs[1, 0].scatter(x_values, campaign_days, color='blue', label='Data')
axs[1, 0].plot(x_values_to_evaluate, campaign_days_fit(x_values_to_evaluate), color='red', label='Cubic Fit')
axs[1, 0].set_title('Campaign Days (Cubic)')
axs[1, 0].set_xlabel('Fraction of Imported Coal')
axs[1, 0].set_ylabel('Campaign Days')
axs[1, 0].legend()
axs[1, 0].text(0.5, 45, f"y = {p_campaign_days[0]:.2f}x^3 + {p_campaign_days[1]:.2f}x^2 + {p_campaign_days[2]:.2f}x + {p_campaign_days[3]:.2f}")

# Shutdown Days plot (Sigmoid Fit)
axs[1, 1].scatter(x_values, shutdown_days, color='blue', label='Data')
axs[1, 1].plot(x_values_to_evaluate, sigmoid_func(x_values_to_evaluate, *params_sigmoid_shutdown), color='red', label='Sigmoid Fit')
axs[1, 1].set_title('Shutdown Days (Sigmoid)')
axs[1, 1].set_xlabel('Fraction of Imported Coal')
axs[1, 1].set_ylabel('Shutdown Days')
axs[1, 1].legend()
axs[1, 1].text(0.5, 8, f"y = {params_sigmoid_shutdown[0]:.2f} / (1 + exp(-{params_sigmoid_shutdown[2]:.2f}(x - {params_sigmoid_shutdown[1]:.2f}))) + {params_sigmoid_shutdown[3]:.2f}")

# Dolochar Generation plot (Cubic Fit)
axs[2, 0].scatter(x_values, dolochar_gen, color='blue', label='Data')
axs[2, 0].plot(x_values_to_evaluate, dolochar_fit(x_values_to_evaluate), color='red', label='Cubic Fit')
axs[2, 0].set_title('Dolochar Generation (Cubic)')
axs[2, 0].set_xlabel('Fraction of Imported Coal')
axs[2, 0].set_ylabel('Dolochar Generation (MT)')
axs[2, 0].legend()
axs[2, 0].text(0.5, 0.28, f"y = {p_dolochar[0]:.2f}x^3 + {p_dolochar[1]:.2f}x^2 + {p_dolochar[2]:.2f}x + {p_dolochar[3]:.2f}")

# Steam Generation plot (Exponential Fit)
axs[2, 1].scatter(x_values, steam_generation, color='blue', label='Data')
axs[2, 1].plot(x_values_to_evaluate, exponential_func_new(x_values_to_evaluate, A_steam, B_steam, C_steam), color='red', label='Exponential Fit')
axs[2, 1].set_title('Steam Generation (Exponential)')
axs[2, 1].set_xlabel('Fraction of Imported Coal')
axs[2, 1].set_ylabel('Steam Generation (MT/hr)')
axs[2, 1].legend()
axs[2, 1].text(0.5, 7, f"y = {A_steam:.2f} + {B_steam:.2f} * exp({C_steam:.2f} * x)")

# Overhead Cost plot (Linear Fit)
axs[3, 0].scatter(x_values, overhead_cost, color='blue', label='Data')
axs[3, 0].plot(x_values_to_evaluate, overhead_fit(x_values_to_evaluate), color='red', label='Linear Fit')
axs[3, 0].set_title('Overhead Cost (Linear)')
axs[3, 0].set_xlabel('Fraction of Imported Coal')
axs[3, 0].set_ylabel('Overhead Cost (INR)')
axs[3, 0].legend()
axs[3, 0].text(0.5, 2000, f"y = {p_overhead[0]:.2f}x + {p_overhead[1]:.2f}")

# Profit Function plot with optimal point
axs[3, 1].plot(x_values_to_evaluate, profits, color='blue', label='Profit Curve')
axs[3, 1].scatter(optimal_x, max_profit, color='red', zorder=5, label=f'Optimal Point\n(x={optimal_x:.2f}, Profit={max_profit:.2f})')
axs[3, 1].set_title('Profit vs. Fraction of Imported Coal')
axs[3, 1].set_xlabel('Fraction of Imported Coal')
axs[3, 1].set_ylabel('Annual Profit (INR)')
axs[3, 1].legend()
axs[3, 1].grid(True, which='both', linestyle='--', linewidth=0.7)
axs[3, 1].axvline(optimal_x, color='red', linestyle='--', lw=1)
axs[3, 1].axhline(max_profit, color='red', linestyle='--', lw=1)

# Display the full plot with all subplots
plt.tight_layout()
plt.show()
