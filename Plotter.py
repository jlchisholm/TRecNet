# Import useful packages
import uproot
import pandas as pd
import awkward as ak
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # need for not displaying plots when running batch jobs
from matplotlib import pyplot as plt
from matplotlib import colors
from sigfig import round

class Particle:
    """ 
    A class for the variables of a particle, and the axis labels and ticks you want to use     
    """
    
    def __init__(self, particle, variables, var_labels, var_units, ticks, tick_labels):
        """
        Initialize: particle name, list of variable names, list of variable axis labels, list of variable units, list of ticks, and list of tick labels
        """

        self.particle = particle
        self.variables = variables
        self.var_labels = dict(zip(variables, var_labels))
        self.var_units = dict(zip(variables, var_units))
        self.ticks = dict(zip(variables, ticks))
        self.tick_labels = dict(zip(variables,tick_labels))


class Dataset:
    """ 
    A class for dataset objects
    """

    def __init__(self, df, reco_method, data_type):
        """
        Initialize: dataframe of data, reconstruction method ('TRecNet' or 'TRecNet+ttbar' or 'KLFitter'), and data type (e.g. 'parton_ejets')
        """

        self.df = df
        self.reco_method = reco_method
        self.data_type = data_type


class Plot:
    """ 
    A class for making various kinds of plots 
    """
    

    def TruthReco_Hist(dataset,particle,variable,save_location,nbins,LL_cut=None):
        """
        Creates and saves a histogram with true and reconstructed data both plotted
        Input: dataset object, particle object, variable, save location, number of bins, and log-likelihood cut (if desired)
        Output: saves histogram as in <save_location> as '<reco_method>_TruthReco_Hist_<data_type>_<particle>_<variable>.png' 
        """

        # Get the dataframe for this dataset
        df = dataset.df

        # Cut on log-likelihood (if desired)
        if dataset.reco_method=='KLFitter' and type(LL_cut)!=None:
            df = df[df['likelihood']>LL_cut]

        # Set the range (based on variable and particle we're plotting)
        ran = particle.ticks[variable][::len(particle.ticks[variable])-1]

        # Plot histograms of true and reco results on the same histogram
        fig, (ax1, ax2) = plt.subplots(nrows=2,sharex=True,gridspec_kw={'height_ratios': [4, 1]})
        reco_n, bins, _ = ax1.hist(df['reco_'+particle.particle+'_'+variable],bins=nbins,range=ran,histtype='step',label='reco')
        truth_n, _, _ = ax1.hist(df['truth_'+particle.particle+'_'+variable],bins=nbins,range=ran,histtype='step',label='truth')
        ax1.legend()

        # Plot the ratio of the two histograms underneath (with a dashed line at 1)
        x_dash,y_dash = np.linspace(min(ran),max(ran),100),[1]*100
        ax2.plot(x_dash,y_dash,'r--')
        bin_width = bins[1]-bins[0]
        ax2.plot(bins[:-1]+bin_width/2, truth_n/reco_n,'ko')  # Using bin width so points are plotting aligned with middle of the bin
        ax2.set(ylim=(0, 2))

        # Set some axis labels
        ax2.set_xlabel(particle.var_labels[variable]+particle.var_units[variable])
        ax1.set_ylabel('Counts')
        ax2.set_ylabel('Ratio (truth/reco)')
        
        # Save the figure as a png in save location
        if dataset.reco_method=='KLFitter' and type(LL_cut)!=None:
            plt.savefig(save_location+dataset.reco_method+'(LL>'+str(LL_cut)+')_TruthReco_Hist_'+dataset.data_type+'_'+particle.particle+'_'+variable,bbox_inches='tight')
            print('Saved Figure: '+dataset.reco_method+'(LL>'+str(LL_cut)+')_TruthReco_Hist_'+dataset.data_type+'_'+particle.particle+'_'+variable)
        else:
            plt.savefig(save_location+dataset.reco_method+'_TruthReco_Hist_'+dataset.data_type+'_'+particle.particle+'_'+variable,bbox_inches='tight')
            print('Saved Figure: '+dataset.reco_method+'_TruthReco_Hist_'+dataset.data_type+'_'+particle.particle+'_'+variable)

        plt.close()

    
    def Normalized_2D(dataset,particle,variable,save_location,color=plt.cm.Blues,LL_cut=None):
        """ 
        Creates and saves a 2D histogram of true vs reconstructed data, normalized across rows
        Input: dataset object, particle object, variable, save location, color scheme, and LL_cut (if desired)
        Output: saves histogram in <save_location> as '<reco_method>_Normalized_2D_<data_type>_<particle>_<variable>.png' 
        """

        # Get the dataframe for this dataset
        df = dataset.df

        # Cut on log-likelihood (if desired)
        if dataset.reco_method=='KLFitter' and type(LL_cut)!=None:
            df = df[df['likelihood']>LL_cut]

        # Define useful constants
        tk = particle.ticks[variable]
        tkls = particle.tick_labels[variable]
        n = len(tk)
        ran = tk[::n-1]

        # Create 2D array of truth vs reco variable (which can be plotted also)
        H, xedges, yedges, im = plt.hist2d(np.clip(df['reco_'+particle.particle+'_'+variable],tk[0],tk[-1]),np.clip(df['truth_'+particle.particle+'_'+variable],tk[0],tk[-1]),bins=tk,range=[ran,ran])

        # Normalize the rows out of 100 and round to integers
        norm = (np.rint((H/np.sum(H.T,axis=1))*100)).T.astype(int)

        # Plot truth vs reco pt with normalized rows
        plt.figure(dataset.data_type+' '+particle.particle+' '+variable+' Normalized 2D Plot')
        masked_norm = np.ma.masked_where(norm==0,norm)  # Needed to make the zero bins white
        plt.imshow(masked_norm,extent=[0,n-1,0,n-1],cmap=color,origin='lower')
        plt.xticks(np.arange(n),tkls,fontsize=9,rotation=-25)
        plt.yticks(np.arange(n),tkls,fontsize=9)
        plt.xlabel('Reco-level '+particle.var_labels[variable]+particle.var_units[variable])
        plt.ylabel('Parton-level '+particle.var_labels[variable]+particle.var_units[variable])
        plt.clim(0,100)
        plt.colorbar()

        # Label the content of each bin
        for j in range (n-1):
            for k in range(n-1):
                if masked_norm.T[j,k] != 0:   # Don't label empty bins
                    plt.text(j+0.5,k+0.5,masked_norm.T[j,k],color="k",fontsize=6,weight="bold",ha="center",va="center")

        # Save the figure in save location as a png
        if dataset.reco_method=='KLFitter' and type(LL_cut)!=None:
            plt.savefig(save_location+dataset.reco_method+'(LL>'+str(LL_cut)+')_Normalized_2D_'+dataset.data_type+'_'+particle.particle+'_'+variable,bbox_inches='tight')
            print('Saved Figure: '+dataset.reco_method+'(LL>'+str(LL_cut)+')_Normalized_2D_'+dataset.data_type+'_'+particle.particle+'_'+variable)
        else:
            plt.savefig(save_location+dataset.reco_method+'_Normalized_2D_'+dataset.data_type+'_'+particle.particle+'_'+variable,bbox_inches='tight')
            print('Saved Figure: '+dataset.reco_method+'_Normalized_2D_'+dataset.data_type+'_'+particle.particle+'_'+variable)

        plt.close()


    def Res(datasets,particle,variable,save_location,nbins,res='Resolution',LL_cuts=None):
        """
        Creates and saves the resolution plots for each variable
        Input: list of dataset objects, particle object, variable, save location, number of bins, 'Resolution' or 'Residuals', and list of cuts on negative LogLikelihood (if any)
        Output: saves histogram in <save_location> as '<res>_<data_type>_<particle>_<variable>.png'
        """

        # Define the absolute range for the resolution plot
        ran = 100 if variable =='pout' else 1   # For some reason pout is quite wide ...

        # Create figure to be filled
        plt.figure(particle.particle+'_'+variable+' '+'Res')
        
        # Fill figure with data
        for dataset in datasets:
            
            # Get dataframe
            df = dataset.df

            # Calculate the resolution (or residuals)
            if res=='Resolution':
                df['res_'+particle.particle+'_'+variable] = (df['reco_'+particle.particle+'_'+variable] - df['truth_'+particle.particle+'_'+variable])/df['truth_'+particle.particle+'_'+variable]
            else:
                df['res_'+particle.particle+'_'+variable] = df['reco_'+particle.particle+'_'+variable] - df['truth_'+particle.particle+'_'+variable]

            # Calculate mean and standard deviation of the resolution
            res_mean = round(df['res_'+particle.particle+'_'+variable].mean(),sigfigs=2)
            res_std = round(df['res_'+particle.particle+'_'+variable].std(),sigfigs=2)

            # Plot the resolution
            plt.hist(df['res_'+particle.particle+'_'+variable],bins=nbins,range=(-ran,ran),histtype='step',label=dataset.reco_method+': No Cuts, $\mu=$'+str(res_mean)+', $\sigma=$'+str(res_std),density=True)

            # Also plot cuts on likelihood if KLFitter (if any)
            if dataset.reco_method=='KLFitter' and type(LL_cuts)==list:
                for cut in LL_cuts:
                    df_cut = df[df['likelihood']>cut]
                    cut_mean = round(df_cut['res_'+particle.particle+'_'+variable].mean(),sigfigs=2)
                    cut_std = round(df_cut['res_'+particle.particle+'_'+variable].std(),sigfigs=2)
                    plt.hist(df_cut['res_'+particle.particle+'_'+variable],bins=nbins,range=(-ran,ran),histtype='step',label=dataset.reco_method+': LL > '+str(cut)+', $\mu=$'+str(cut_mean)+', $\sigma=$'+str(cut_std),density=True)
            
        # Add some labels
        plt.legend(prop={'size': 6})
        plt.xlabel(particle.var_labels[variable]+' '+res)
        plt.ylabel('Counts')

        # Save figure in save location
        plt.savefig(save_location+res+'_'+dataset.data_type+'_'+particle.particle+'_'+variable,bbox_inches='tight')
        print('Saved Figure: '+res+'_'+dataset.data_type+'_'+particle.particle+'_'+variable)

        plt.close()


    def Res_vs_Var(datasets,particle,y_var,x_var,x_bins,save_location,y_res='Resolution',LL_cuts=None):
        """
        Creates and saves the resolution vs variable plots
        Input: datasets (could be multiple), particle, res y-variable, res x-variable, [first bin edge, last bin edge, width of bins], save location, resolution vs residuals for y, and negative log-likelihood cuts (list) if any 
        Output: saves histogram as '<y_var>_<y_res>_vs_<x_var>_<data_type>_<particle>.png'
        """

        for dataset in datasets:

            # Get dataframe
            df = dataset.df

            # Calculate the resolution (or residuals)
            if y_res=='Resolution':
                df['res_'+particle.particle+'_'+y_var] = (df['reco_'+particle.particle+'_'+y_var] - df['truth_'+particle.particle+'_'+y_var])/df['truth_'+particle.particle+'_'+y_var]
            else:
                df['res_'+particle.particle+'_'+y_var] = df['reco_'+particle.particle+'_'+y_var] - df['truth_'+particle.particle+'_'+y_var]
                
            # Create a scatter plot to be filled
            plt.figure(y_var+' '+y_res+' vs '+x_var)

            points = []   # Array to hold var vs fwhm values
            for bottom_edge in np.arange(x_bins[0],x_bins[1],x_bins[2]):

                # Set some helpful variables
                top_edge = bottom_edge+x_bins[2]
                middle = bottom_edge+(x_bins[2]/2)

                # Look at resolution at a particular value of var
                cut_temp = df[df['truth_'+particle.particle+'_'+x_var]>=bottom_edge]      # Should I fold in edges of first and last?
                cut_temp = cut_temp[cut_temp['truth_'+particle.particle+'_'+x_var]<top_edge]

                # Get standard deviations with dataframe method
                sigma = cut_temp['res_'+particle.particle+'_'+y_var].std()
                
                # Add point to list
                points.append([middle,sigma])

            # Set color index
            #if dataset.reco_method=='KLFitter':
            #    i = 0
            #else:
            #    i = 4

            # Put data in the scatterplot
            #plt.scatter(np.array(points)[:,0],np.array(points)[:,1],color=plt.cm.tab20c(i),label=dataset.reco_method+': No Cuts')
            plt.scatter(np.array(points)[:,0],np.array(points)[:,1],label=dataset.reco_method+': No Cuts')

            # Reset color index for LL cuts
            #i = 1

            # Also make cuts on likelihood of KLFitter (if any)
            if dataset.reco_method=='KLFitter' and type(LL_cuts)==list:
                for cut in LL_cuts:
               
                    # Make the LL cut
                    df_cut = df[df['likelihood']>cut]

                    points = []   # Array to hold var vs fwhm values
                    for bottom_edge in np.arange(x_bins[0],x_bins[1],x_bins[2]):
                        
                        # Set some helpful variables
                        top_edge = bottom_edge+x_bins[2]
                        middle = bottom_edge+(x_bins[2]/2)

                        # Look at resolution at a particular value of var
                        cut_temp = df_cut[df_cut['truth_'+particle.particle+'_'+x_var]>=bottom_edge]      # Should I fold in edges of first and last?
                        cut_temp = cut_temp[cut_temp['truth_'+particle.particle+'_'+x_var]<top_edge]
                        
                        # Get standard deviations
                        sigma = cut_temp['res_'+particle.particle+'_'+y_var].std()

                        # Add point to list
                        points.append([middle,sigma])

                    # Put data in the scatterplot
                    #plt.scatter(np.array(points)[:,0],np.array(points)[:,1],color=plt.cm.tab20c(i),label=dataset.reco_method+': LL > '+str(cut))
                    plt.scatter(np.array(points)[:,0],np.array(points)[:,1],label=dataset.reco_method+': LL > '+str(cut))

                    # Increase color index
                    #i+=1

        # Add some labels
        plt.xlabel('Parton-level '+particle.var_labels[x_var]+particle.var_units[x_var])
        plt.ylabel('$\sigma$ of '+particle.var_labels[y_var]+' '+y_res)
        plt.legend()

        # Save figure in save location
        plt.savefig(save_location+y_var+'_'+y_res+'_vs_'+x_var+'_'+dataset.data_type+'_'+particle.particle,bbox_inches='tight')
        print('Saved Figure: '+y_var+'_'+y_res+'_vs_'+x_var+'_'+dataset.data_type+'_'+particle.particle)
        
        plt.close()
