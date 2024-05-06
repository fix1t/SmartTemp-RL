import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.dates as mdates
import os

def plot_all_in_one(outside_temp, inside_temp, occupancy, heater_status, time,
                    target_min=21, target_max=23, output_dir='out/plots', name='all_in_one_plot'):
    # Convertions:
    target_min = float(target_min)
    target_max = float(target_max)
    # Convert arrays to numpy arrays with appropriate types
    outside_temp = np.array(outside_temp, dtype=float)
    inside_temp = np.array(inside_temp, dtype=float)
    heater_status = np.array(heater_status, dtype=float)
    occupancy = np.array(occupancy, dtype=bool)
    # Convert datetime objects to float days
    if isinstance(time[0], datetime):
        time = mdates.date2num(time)

    # Prepare the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %y-%m-%d'))

    # Plotting temperature data
    ax.plot(time, outside_temp, label='Outside Temperature', color='grey')
    ax.plot(time, inside_temp, label='Inside Temperature', color='black')

    y_limit = [np.min(outside_temp) - 2, np.max(inside_temp) + 2]
    x_limit = [min(time), max(time)]

    # Highlighting occupancy and comfort range based on boolean array

    # PPO
    # ax.fill_between(time, y_limit[0], y_limit[1], where=occupancy, color='lightblue', alpha=0.8, label='Occupancy')
    # ax.fill_between(time, target_min, target_max, where=occupancy, color='dodgerblue', alpha=0.8, label='Target Temp (Occupied)')
    # DQL
    ax.fill_between(time, y_limit[0], y_limit[1], where=occupancy, color='moccasin', alpha=0.8, label='Occupancy')
    ax.fill_between(time, target_min, target_max, where=occupancy, color='darkorange', alpha=0.8, label='Target Temp (Occupied)')

    # Setting up heater status color gradient line
    cmap = LinearSegmentedColormap.from_list("heater_cmap", ["white", "red"])
    norm = plt.Normalize(0, 5)
    heater_values = np.full_like(time, (np.max(inside_temp) + np.min(outside_temp)) / 2, dtype=float)

    points = np.array([time, heater_values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=6)
    lc.set_array(heater_status)
    ax.add_collection(lc)

    # Additional plot settings
    ax.set_xlabel('Time',fontsize=14)
    ax.set_ylabel('Temperature (°C)',fontsize=14)
    ax.legend(bbox_to_anchor = (1, -0.01), loc='upper right', fontsize=12)
    ax.set_ylim(y_limit)
    ax.set_xlim(x_limit)

    plt.gcf().autofmt_xdate()  # Format date labels for readability
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label('Heater Status', fontsize=12)
    plt.tight_layout()

    # Save and show
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{name}.png')
    plt.close()

def plot_all_in_one_rewards(time, occupancy, heater_status, rewards,
                    output_dir='out/plots', name='all_in_one_plot_rew', threshold=0):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert occupancy to a numpy boolean array
    occupancy = np.array(occupancy, dtype=bool)

    # Convert datetime objects to float days if needed
    if isinstance(time[0], datetime):
        time = mdates.date2num(time)

    # Prepare the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %y-%m-%d'))

    # Plotting reward data
    ax.plot(time, rewards, label='Rewards', color='green')

    y_limit = [min(rewards) - 1, max(rewards) + 1]

    # Highlighting occupancy based on boolean array
    # PPO
    # ax.fill_between(time, y_limit[0], y_limit[1], where=occupancy, color='lightblue', alpha=0.8, label='Occupancy')
    # DQL
    ax.fill_between(time, y_limit[0], y_limit[1], where=occupancy, color='moccasin', alpha=0.8, label='Occupancy')

    # Heater status with a color gradient
    cmap = LinearSegmentedColormap.from_list("heater_cmap", ["white", "red"])
    norm = plt.Normalize(0, 5)
    heater_values = np.full_like(time, (y_limit[0]+y_limit[1])//2, dtype=float)  # Mid-line for heater status visualization

    points = np.array([time, heater_values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=6)
    lc.set_array(heater_status)
    ax.add_collection(lc)

    # Create gradient fill beneath threshold
    gradient_fill(ax, time, rewards, threshold, upper_color='green', lower_color='red')

    # Additional plot settings
    ax.set_xlabel('Time',fontsize=14)
    ax.set_ylabel('Rewards',fontsize=14)
    ax.legend(bbox_to_anchor=(1, -0.01), loc='upper right', fontsize=12)
    ax.set_ylim(y_limit)  # Adjust y-limits based on reward range
    ax.set_xlim([min(time), max(time)])

    plt.gcf().autofmt_xdate()  # Format date labels for readability
    cb = plt.colorbar(lc, ax=ax)
    cb.set_label('Heater Status', fontsize=12)
    plt.tight_layout()

    # Save and show
    plt.savefig(f'{output_dir}/{name}.png')
    plt.close()

def gradient_fill(ax, x, y, threshold, upper_color, lower_color):
    # Ensure the y array has the correct length as x for fill_between
    y = np.array(y)

    # Create a gradient from red at the bottom to transparent at the threshold
    z = np.empty((100, 4), dtype=float)
    rgb = plt.cm.Reds(np.linspace(0, 1, 100))[:, :3]  # Get the red color palette
    z[:, :3] = rgb
    z[:, 3] = np.linspace(0, 1, 100)  # Transparency gradient

    # Creating a custom colormap
    gradient_cmap = ListedColormap(z)

    # Define the gradient fill below the threshold
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Fill the area above the threshold with a solid color
    ax.fill_between(x, threshold, ymax, color=upper_color, alpha=0.3)

    # Gradient fill below the threshold
    # Prepare a PolyCollection filling the entire area under the curve, fading to the bottom
    verts = [(xmin, ymin), *zip(x, y), (xmax, ymin)]
    poly = PolyCollection([verts], facecolor=gradient_cmap(np.linspace(0, 1, 100)), edgecolor='none')
    ax.add_collection(poly)

def plot_variance(all_data):
    # Combine all data into a 2D numpy array for easier manipulation
    all_data = np.vstack(all_data)

    # Calculate min, max, and mean across the rows for each column (i.e., each timestep)
    min_scores = np.min(all_data, axis=0)
    max_scores = np.max(all_data, axis=0)
    mean_scores = np.mean(all_data, axis=0)

    # Creating the plot
    plt.figure(figsize=(12, 8))

    # Plotting the mean score with a smoother and thicker line
    plt.plot(mean_scores, color='darkorange', label='Average Score', linestyle='-', linewidth=2.5)

    # Shading the area between the min and max score with a gradient
    plt.fill_between(range(len(min_scores)), min_scores, max_scores, color='royalblue', alpha=0.8, label='Min-Max Range')

    # Enhancing the graph aesthetics
    # plt.title('Score Variability Across Environments Per Episode', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a legend with a shadow
    legend = plt.legend(frameon=True, bbox_to_anchor = (1.15, 0), loc='upper right', fontsize=12)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_boxstyle('round,pad=0.5')

    plt.grid(True, linestyle='--', alpha=0.6)  # Adding a styled grid
    plt.tight_layout()
    plt.show()
    plt.close()
