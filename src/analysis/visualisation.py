#Top level visualisation functions 
from typing import Dict, List, Literal, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from dataclasses import dataclass
import math

DEFAULT_QUANTILES = [0.0275, 0.975]

def plot_credint(
        data: Dict[str, pd.DataFrame], 
        title: str = 'Estimated Mean and Credible Intervals Across Posterior Estimation Methods',
        quantiles: List[float] = DEFAULT_QUANTILES, 
        figsize = (10, 10), 
    ) -> plt.Figure: 
    '''
    Plot credible intervals and means for multiple posterior estimation methods across common estimated parameters.

    Args: 
        data: Dict mapping method names to dataframes 
        title: Plot title 
        quantiles: [Lower, Upper] quantile 
        figsize: Figure size

    '''
    variables = sorted(list(set.intersection(*[set(df.columns) for df in data.values()])))

    #Calculate layout 
    n_vars = len(variables)
    n_cols = 2 
    n_rows = n_rows = math.ceil(n_vars/n_cols)

    #Create figure and axes 
    fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize, constrained_layout = True)
    fig.suptitle(title, fontsize = 14)

    #Delete unused axes 
    flattened_axes = axes.flatten()
    for ax in flattened_axes[n_vars:]:
        ax.remove()

    #Calculate relevant statistics
    stats = {}
    colors = plt.cm.Set1(range(len(data))) #Auto-generate colours 
    for i, (method, df) in enumerate(data.items()): 
        stats[method] = {
            'lower': df.quantile(quantiles[0]), 
            'upper': df.quantile(quantiles[1]), 
            'mean': df.mean(), 
            'color': colors[i]
        }

    #Initiate legends

    #Plot 
    for i, var in enumerate(variables): 
        #Focus on the ith model parameter 
        ax = flattened_axes[i]
        for j, method in enumerate(data.keys()): 
            df = data[method]
            lower = stats[method]['lower'][var]
            upper = stats[method]['upper'][var]
            mean = stats[method]['mean'][var]
            color = stats[method]['color']
        
            ax.plot([lower, upper], [j, j], color = color, linewidth = 8, alpha = 0.6)
            ax.plot(mean, j, 'ko', markersize = 6)

        #Add MCMC reference line, if it exists 
        if 'MCMC' in data:
            ax.axvline(stats['MCMC']['mean'][var], color = 'red', linestyle = '--', alpha = 0.8)
            
        
        #Set title 
        ax.set_title(var)
        ax.set_yticks(range(len(data)))
        ax.grid(False)

    #Add legends 
    #Legend - Posterior Estimation Method 
    handles = [mpatches.Patch(color = stats[method]['color'], label = "Credible Interval: " + method ) for method in stats.keys()]
    handles.append(mlines.Line2D([], [], color = 'black', marker = 'o', markersize = 6, label = 'Estimated Posterior Mean'))
    if 'MCMC' in data: 
        handles.append(mpatches.Patch(color='red', linestyle='--', label='Posterior Mean: MCMC Reference'))
    fig.legend(handles = handles, loc = 'lower right', title = 'Plot Elements', ncols = 1)
