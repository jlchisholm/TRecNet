######################################################################
#                                                                    #
#  Plots.py                                                          #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.10/25                                                #
#                                                                    #
#  Defines a plotting class with functions for plotting truth vs.    #
#  reco histograms, confusion matrices, systematics histograms,      #
#  resolution histograms, and plots of resolution as a function of   #
#  a specified variable. Intended for visualizing and comparing the  #
#  results of different ttbar reconstruction methods.                # 
#                                                                    #
#  Thoughts for improvements: Include systematics, allow pt cuts     #
#  for truth vs reco and confusion matrices, maybe better color      #
#  scheme handling (e.g. with cuts on the data).                     #
#                                                                    #
######################################################################


# Import useful packages
import numpy as np
import matplotlib
matplotlib.use('Agg')  # need for not displaying plots when running batch jobs
from matplotlib import pyplot as plt
from matplotlib import colors
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from scipy.stats import norm
from scipy.stats import cauchy
from sigfig import round
import Util


PLOT_TYPES = ['TruthReco','CM','Res','ResVsVar','Sys']



def TruthReco_Hist(dataset,particle,observable,x_min,x_max,nbins=30,save_loc='./'):
    """
    Creates and saves a histogram with true and reconstructed data both plotted, for a given dataset, particle, and observable.

        Parameters:
            dataset (Dataset object): Dataset object with the data you want to plot.
            particle (Particle object): Particle object of the particle you want to plot.
            observable (Observable object): Observable object of the observable you want to plot.
            x_min (int or float): Minimum value to plot.
            x_max (int or float): Maximum value to plot.

        Options:
            nbins (int): Number of desired bins for the histogram (default: 30).
            save_loc (str): Directory where you want the histogram saved to (default: current directory).

        Returns:
            Saves histogram in <save_loc> as '<reco_method>_TruthReco_Hist_<data_type>_<particle>_<observable>.png' .
    """



    # Define a useful string
    name = particle.name+'_'+observable.name

    # Plot histograms of true and reco results on the same histogram
    _, (ax1, ax2) = plt.subplots(nrows=2,sharex=True,gridspec_kw={'height_ratios': [4, 1]})
    truth_n, _, _ = ax1.hist(dataset.df['truth_'+name],bins=nbins,range=(x_min,x_max),histtype='step',label='truth',color='black')
    reco_n, bins, _ = ax1.hist(dataset.df['reco_'+name],bins=nbins,range=(x_min,x_max),histtype='step',label='reco',color=dataset.color)
    
    # Make sure histograms aren't getting cut off
    max_truth = max(truth_n)*1.05
    max_reco = max(reco_n)*1.05
    ax1.set(ylim=(0,max([max_truth,max_reco])))

    # Plot the ratio of the two histograms underneath (with a dashed line at 1)
    x_dash,y_dash = np.linspace(x_min,x_max,100),[1]*100
    ax2.plot(x_dash,y_dash,'k--')
    bin_width = np.diff(bins)
    ax2.plot(bins[:-1]+bin_width/2, truth_n/reco_n,'o',color=dataset.color)  # Using bin width so points are plotted aligned with middle of the bin
    ax2.set(ylim=(0, 2))
    #ax2.set_yscale('log')

    # Set some axis labels
    ax2.set_xlabel(particle.labels[observable.name])
    ax1.set_ylabel('Counts')
    ax2.set_ylabel('Ratio (truth/reco)')
    ax1.legend()
    
    # Save the figure as a png in save location
    fig_name = dataset.reco_method+'('+dataset.cut_tag+')_TruthReco_Hist_'+name
    plt.savefig(save_loc+fig_name+'.png',bbox_inches='tight')
    print('Saved Figure: '+fig_name)

    plt.close()
    

def Confusion_Matrix(dataset,particle,observable,ticks,tick_labels,norm=True,tag='',save_loc='./'):
    """ 
    Creates and saves a 2D histogram of true vs reconstructed data, normalized across rows, for a given dataset, particle, and observable.

        Parameters:
            dataset (Dataset object): Dataset object with the data you want to plot.
            particle (Particle object): Particle object of the particle you want to plot.
            observable (Observable object): Observable object of the observable you want to plot.
            ticks (array): Array of the bin edges (int or float, depending on the observable).
            tick_labels (list of str): List of labels for the ticks.  
        
        Options:
            norm (bool): Whether or not to normalize the confusion matrix across rows (default: True).
            tag (str): Extra tag to add to the plot save name.
            save_loc (str): Directory where you want the histogram saved to (default: current directory).

        Returns:
            Saves histogram in <save_loc> as '<reco_method>_Confusion_Matrix_<data_type>_<particle>_<observable>.png'. 
    """
    
    # Make the appropriate color map
    color_map=colors.LinearSegmentedColormap.from_list('my_cmap', ['white', dataset.color])

    # Define a useful string and some important constants
    name = particle.name+'_'+observable.name
    n = len(ticks)
    ran = ticks[::n-1]

    # Create 2D array of truth vs reco observable (which can be plotted also)
    H, _, _, _ = plt.hist2d(np.clip(dataset.df['reco_'+name],ticks[0],ticks[-1]),np.clip(dataset.df['truth_'+name],ticks[0],ticks[-1]),bins=ticks,range=[ran,ran])

    # Normalize across rows (if desired)
    if norm:
        H = np.divide(H,np.sum(H,axis=0),where=np.sum(H,axis=0)!=0)  # This should ensure we're not dividing by zero
        H = H*100
        #H = (H/np.sum(H,axis=0))*100  # Old way
    
    # Round to integers (and transpose, so it's where we need it for plotting later)
    cm = np.rint(H).T.astype(int)

    # Plot truth vs reco pt with normalized rowsx
    plt.figure(particle.name+' '+observable.name+' Normalized 2D Plot')
    masked_cm = np.ma.masked_where(cm==0,cm)  # Needed to make the zero bins white
    plt.imshow(masked_cm,extent=[0,n-1,0,n-1],cmap=color_map,origin='lower')
    plt.xticks(np.arange(n),tick_labels,fontsize=12,rotation=-25)
    plt.yticks(np.arange(n),tick_labels,fontsize=12)
    plt.xlabel('Reco-level '+particle.labels[observable.name], fontsize=15)
    plt.ylabel('Parton-level '+particle.labels[observable.name], fontsize=15)
    if norm: plt.clim(0,100)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)

    # Label the content of each bin
    for j in range (n-1):
        for k in range(n-1):
            if masked_cm.T[j,k] != 0:   # Don't label empty bins
                plt.text(j+0.5,k+0.5,masked_cm.T[j,k],color='k',fontsize=10,weight="bold",ha="center",va="center")

    # Save the figure in save location as a png
    fig_name = dataset.reco_method+'('+dataset.cut_tag+')_Confusion_Matrix'+tag+'_'+name
    plt.savefig(save_loc+fig_name+'.png',bbox_inches='tight')
    print('Saved Figure: '+fig_name)

    plt.close()


def Res_Hist(datasets,particle,observable,save_loc='./',tag='',nbins=30,include_moments=False):
    """
    Creates and saves a resolution (or residual) plot all datasets provided, for a given particle and observable.

        Parameters:
            datasets (list of Dataset objects): RecoModel objects of the data you want to plot.
            particle (Particle object): Particle object of the particle you want to plot.
            observable (Observable object): Observable object of the observable you want to plot.

        Options:
            save_loc (str): Directory where you want the histogram saved to (default: current directory).
            tag (str): Extra tag to add to the plot save name.
            nbins (int): Number of desired bins for the histogram (default: 30).
            include_moments (bool): Whether or not to include the mean and standard deviation in the legend.

        Returns:
            Saves histogram in <save_loc> as '<res>_<data_type>_<particle>_<observable>.png'.
    """

    # Define a useful string
    name =particle.name+'_'+observable.name

    # Create figure to be filled
    plt.figure(name+' '+'Res')
    
    # Get percentage of the dataset's events in the range of the plot
    num_events = {}
    num_in_events = {}
    in_dic = {}
    
    # Fill figure with data
    for dataset in datasets:
        
        # Get dataframe
        df = dataset.df

        # Calculate the resolution (or residuals)
        df = Util.calculate_res(particle,observable,df)

        # Calculate mean and standard deviation of the resolution
        if include_moments==True:
            res_mean = round(df['res_'+name].mean(),sigfigs=2)
            res_std = round(df['res_'+name].std(),sigfigs=2)
            fit_mean = round(df['res_'+name][df['res_'+name]>-1][df['res_'+name]<1].mean(),sigfigs=2)
            fit_std = round(df['res_'+name][df['res_'+name]>-1][df['res_'+name]<1].std(),sigfigs=2)
            mom_tag = '\n'+r'$\mu_{\mathrm{total}}=$'+str(res_mean)+', '+r'$\sigma_{\mathrm{total}}=$'+str(res_std)+',\n'+r'$\mu_{\mathrm{core}}=$'+str(fit_mean)+', '+r'$\sigma_{\mathrm{core}}=$'+str(fit_std)
        else:
            mom_tag = ''

        # Plot the resolution
        #model_label = dataset.reco_method_short+': '+dataset.cut_tag+' ('+str(dataset.perc_events)+'%)'+mom_tag
        model_label = dataset.reco_method_short+'('+dataset.cut_tag+')'+mom_tag if dataset.cut_tag!='No Cuts' else dataset.reco_method_short+mom_tag
        plt.hist(df['res_'+name],bins=nbins,range=(-1,1),histtype='step',label=model_label,density=True,color=dataset.color)
        
        # Get percentage of dataset's events in the plot
        in_events = df['res_'+name][df['res_'+name]>-1]
        in_events = df['res_'+name][df['res_'+name]<1]
        num_events[dataset.reco_method_short+'('+dataset.cut_tag+')'] = len(df['res_'+name])
        num_in_events[dataset.reco_method_short+'('+dataset.cut_tag+')'] = len(in_events)
        in_dic[dataset.reco_method_short+'('+dataset.cut_tag+')'] = (len(in_events)/len(df['res_'+name]))*100

    # Add some labels
    plt.legend(prop={'size': 9})
    #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10},borderaxespad=0)
    plt.xlabel(particle.labels_nounits[observable.name]+' '+observable.res, fontsize=14)
    plt.ylabel('Events (Normalized)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save figure in save location
    fig_name = observable.res+'_'+name+tag
    plt.savefig(save_loc+fig_name+'.png',bbox_inches='tight')
    print('Saved Figure: '+fig_name)

    plt.close() 
    
    # Save crucial plot info
    Util.save_plot_info(save_loc, fig_name, num_events, num_in_events, in_dic)


def Res_vs_Var(datasets,particle,y_obs,x_obs,ticks,tick_labels,save_loc='./',tag='',core_fit='nofit'):
    """
    Creates and saves a plot of resolution (or residual) for a given observable against another (or the same) given observable, for all datasets provided, for a given particle.

        Parameters:
            datasets (list of Dataset objects): List of dataset objects of the datasets you want to plot.
            particle (Particle object): Particle object of the particle you want to plot.
            y_var (Observable object): Observable whose resolution (or residuals) will be plotted on the y-axis.
            x_var (Observable object): Observable whose parton level values will be plotted on the x-axis.
            ticks (array): Array of the bin edges (int or float, depending on the observable).
            tick_labels (list of str): List of labels for the ticks.

        Options:
            save_loc (str): Directory where you want the histogram saved to (default: current directory).
            tag (str): Extra tag to add to the plot save name.
            core_fit (str): Type of fit you want to use for the width calculations (default: 'nofit', other options: 'gaussian' or 'cauchy').

        Returns:
            Saves histogram in <save_loc> as '<y_obs>_<y_res>_vs_<x_obs>_<data_type>_<particle>.png'.
    """
    
    # Useful to define
    yfocus = particle.name+'_'+y_obs.name
    y_res = y_obs.res
    
    # Create a scatter plot to be filled
    plt.figure(y_obs.name+' '+y_res+' vs '+x_obs.name)
    
    for dataset in datasets:
        
        # Get dataframe
        df = dataset.df
        
        # Calculate the resolution (or residuals)
        df = Util.calculate_res(particle,y_obs,df)
        
        # Get data points for histogram (going through each bin here)
        points = []   # Array to hold var vs fwhm values
        for i, bottom_edge in enumerate(ticks[:-1]):

            # Set some helpful observables
            top_edge = ticks[i+1]
            middle = bottom_edge + (top_edge - bottom_edge)/2

            # Look at resolution at a particular value of var
            cut_temp = df[df['truth_'+particle.name+'_'+x_obs.name]>=bottom_edge]      # Should I fold in edges of first and last?
            cut_temp = cut_temp[cut_temp['truth_'+particle.name+'_'+x_obs.name]<top_edge]

            # Get standard deviations
            if core_fit=='gaussian':
                _, sigma = norm.fit(cut_temp['res_'+yfocus][cut_temp['res_'+yfocus]>-1][cut_temp['res_'+yfocus]<1])
                #sigma = cut_temp['res_'+yfocus][cut_temp['res_'+yfocus]>-1][cut_temp['res_'+yfocus]<1].std()
            elif core_fit=='cauchy':
                _, sigma = cauchy.fit(cut_temp['res_'+yfocus][cut_temp['res_'+yfocus]>-1][cut_temp['res_'+yfocus]<1])
            else:
                sigma = cut_temp['res_'+yfocus].std()
                
            # Add point to list
            points.append([middle,sigma])

        # Plot the data
        xpoints = np.array(range(len(points)))+0.5
        ypoints = np.array(points)[:,1]
        xerror = np.full(len(points),0.5)   
        #model_label = dataset.reco_method_short+': '+dataset.cut_tag+' ('+str(dataset.perc_events)+'%)'
        model_label = dataset.reco_method_short+'('+dataset.cut_tag+')' if dataset.cut_tag!='No Cuts' else dataset.reco_method_short
        plt.errorbar(xpoints, ypoints,xerr=xerror,label=model_label,color=dataset.color, fmt='o')

    # Add some labels
    plt.xlabel('Parton-level '+particle.labels[x_obs.name], fontsize=14)
    y_sigma_str = r'$\sigma_{\mathrm{core}}$' if core_fit=='gaussian' else r'$\sigma_{\mathrm{total}}$'
    plt.ylabel(y_sigma_str+' of '+particle.labels_nounits[y_obs.name]+' '+y_res, fontsize=14)
    plt.legend(prop={'size': 12})
    plt.xticks(np.arange(len(ticks)),tick_labels,fontsize=12)
    plt.yticks(fontsize=12)

    # Save figure in save location
    tag = tag+'_' if tag!='' else tag
    fig_name = y_obs.name+'_'+y_res+'_vs_'+x_obs.name+'_'+tag+particle.name
    plt.savefig(save_loc+fig_name+'.png',bbox_inches='tight')
    print('Saved Figure: '+fig_name)
    
    plt.close()
    
    
    
def Sys_Hist(datasets,particle,observable,ticks,tick_labels,save_loc='./',tag=''):
    """
    Creates and saves a histogram of the systematics for all datasets provided, for a given particle and observable.

        Parameters:
            datasets (list of Dataset objects): List of dataset objects of the datasets you want to plot.
            particle (Particle object): Particle object of the particle you want to plot.
            observable (observable object): Observable object of the observable you want to plot.
            ticks (array): Array of the bin edges (int or float, depending on the observable).
            tick_labels (list of str): List of labels for the ticks.  

        Options:
            save_loc (str): Directory where you want the histogram saved to (default: current directory).
            tag (str): Extra tag to add to the plot save name.

        Returns:
            Saves histogram in <save_loc> as '<res>_<data_type>_<particle>_<observable>.png'.                
    """

    # Define a useful things
    name = particle.name+'_'+observable.name
    n = len(ticks)
    ran = ticks[::n-1]

    # Go through and plot each of the datasets
    for dataset in datasets:

        # Create a temporary plot to bin the data (set density=True to normalize the counts)
        plt.figure('Temporary')
        reco_n, bins, _ = plt.hist(np.clip(dataset.df['reco_'+name],ticks[0],ticks[-1]),bins=ticks,range=ran,density=True)
        sysUP_n, _, _ = plt.hist(np.clip(dataset.sysUP_df['reco_'+name],ticks[0],ticks[-1]),bins=ticks,range=ran,density=True)
        sysDOWN_n, _, _ = plt.hist(np.clip(dataset.sysDOWN_df['reco_'+name],ticks[0],ticks[-1]),bins=ticks,range=ran,density=True)
        plt.close('Temporary')

        # Calculate the up and down fractional uncertainties
        sysUP_results = np.array([100*(up-nom)/nom for up,nom in zip(sysUP_n,reco_n)])
        sysDOWN_results = np.array([100*(down-nom)/nom for down,nom in zip(sysDOWN_n,reco_n)])

        # Switch back to the systematics figure
        plt.figure('Sys')
        #plt.hist(bins[:-1], bins, weights=sysUP_results, histtype='step', color=dataset.color, linestyle='dotted')
        #plt.hist(bins[:-1], bins, weights=sysDOWN_results, histtype='step', color=dataset.color, label=dataset.reco_method+': '+dataset.cut_tag)

        # Need to sort systematics into which one is on top and which one is on bottom -- if both are on the same side of zero, just use the bigger one
        pos_weights = np.array([max(up,down) if max(up,down)>0 else 0 for up, down in zip(sysUP_results,sysDOWN_results)])
        neg_weights = np.array([min(up,down) if min(up,down)<0 else 0 for up, down in zip(sysUP_results,sysDOWN_results)])

        # Plot the fractional uncertainties
        plt.hist(np.arange(n-1), bins=np.arange(n), weights=pos_weights, histtype='step', color=dataset.color, label=dataset.reco_method+': '+dataset.cut_tag+' ('+str(dataset.perc_events)+'%)')
        plt.hist(np.arange(n-1), bins=np.arange(n), weights=neg_weights, histtype='step', color=dataset.color)

    # Draw a dashed line at zero
    x_dash, y_dash = np.linspace(0,n-1,n-1),[0]*(n-1)
    plt.plot(x_dash,y_dash,'k--')

    # Set some axis labels
    plt.xlabel(particle.labels[observable.name])
    plt.ylabel('Fractional Uncertainty [%]')
    plt.xticks(np.arange(n),tick_labels,fontsize=12,rotation=-25)
    plt.yticks(fontsize=12)
    plt.legend(prop={'size': 6})

    # Save the figure as a png in save location
    tag = tag+'_' if tag!='' else tag
    fig_name = 'Systematics_'+tag+name
    plt.savefig(save_loc+fig_name+'.png',bbox_inches='tight')
    print('Saved Figure: '+fig_name)

    plt.close()