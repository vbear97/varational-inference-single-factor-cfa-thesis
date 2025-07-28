#Top level visualisation functions 
from typing import Dict, List, Literal, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from dataclasses import dataclass
import math
import seaborn as sns
import numpy as np 

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
    subtitle = f'Credible Interval Quantiles: {quantiles}'
    fig.suptitle(title + '\n' + subtitle, fontsize = 14)

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

def plot_eta(vidf: pd.DataFrame, mcmc_df: pd.DataFrame, moment = Literal['mean', 'var'], figsize = (5,5)): 
    '''Create scatterplot of eta moments (y-axis) vs MCMC means'''
    
    fig, ax = plt.subplots(figsize = (5,5))
    fig.suptitle(f'Comparison of Eta {moment}')
    #Plot 
    ax.scatter(x = getattr(vidf, moment)(), y = getattr(mcmc_df, moment)())
    ax.set_ylabel(f'MCMC Eta {moment}')
    ax.set_xlabel(f'VI Eta {moment}')
    ax.axline(xy1 = (0,0), slope = 1, label = 'y=x line')

def plot_moments_boxplot(
    batched_vi_models_by_alpha_k: Dict[tuple[float], list[pd.DataFrame]],
    mcmc_df: pd.DataFrame, 
    moment: Literal['mean', 'var'], 
    showoutliers = True, 
    fontsize: int = 20
    ): 

    #Create VI statistics
    list_summary_df: list[pd.Series] = []
    for (alpha,K), list_df in batched_vi_models_by_alpha_k.items(): 
        for index, df in enumerate(list_df):
            moments= pd.DataFrame([getattr(df, moment)], name = 'moment')
            moments['alpha'] = alpha 
            moments['K'] = K 
            moments['S'] = index 
            list_summary_df.append(moments)
    
    #assume same length 
    S = len(list_df)
    df = pd.concatenate(list_summary_df, ignore_index = True)

    #Create MCMC statistics 
    mcmc_summary = getattr(mcmc_df, moment)
    
    #Boxplot 
    g = sns.catplot(data  = df, 
                    x = 'alpha', 
                    y = 'moment', 
                    hue = 'K', 
                    col = 'parameter',
                    kind = 'box', 
                    showmeans = True, 
                    sharex = False, 
                    sharey = False,
                    col_wrap = 3, 
                    height = 9,
                    showfliers = showoutliers
                    )
    
    #Add MCMC reference lines
    for parameter, ax in g.axes_dict.items(): 
        y = mcmc_summary[parameter]
        #Add reference line 
        ax.axhline(y = y, color = 'red', ls = '--')
        #Print out y value with annotation 
        ax.text(x = 3.5, y = y, s = f'MCMC {moment} =' + str(np.round(y,3)), fontdict = {'fontstyle': 'normal', 'fontsize': fontsize, 'color': 'red', 'ha': 'right'})
        ax.set_xlabel('alpha', fontsize = fontsize)
        ax.set_ylabel('Variational ' + moment, fontsize = fontsize)
        #make axes numbers bigger
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.set_title(parameter)
        ax.title.set_size(fontsize)
    
    #formatting 
    ##Legend 
    sns.move_legend(g, "center right", fontsize= 20, title_fontsize = 20)
    #Attach MCMC details 
    g._legend.legendHandles.append(mpatches.Patch(color='red', linestyle='--', label='Posterior Mean: MCMC Reference'))

    #adjust spacing within subplots
    g.fig.subplots_adjust(
                    left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.4)

    #title 
    g.figure.suptitle(f'Spread of VR-alpha estimated posterior {moment}s, (per alpha, K configuration for S = {S}', fontsize = 30, y = 0.95, wrap = True)
                   
    return g