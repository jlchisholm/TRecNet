# Import useful packages
import uproot
import pandas as pd
import awkward as ak
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # need for not displaying plots when running batch jobs
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from scipy.stats import norm
from scipy.stats import cauchy
from sigfig import round
from analysis import Analysis


class Variable:
    """
    Variable object class (for plotting purposes)
    """

    def __init__(self, name, label, ticks, unit='',alt_names=[]):
        """
        Initializes a variable object.

            Parameters:
                name (str): Name of the variable (e.g.'pt')
                label (str): Label for the variable (note: this may be different than the name if you want to, for example, use Latex formatting).
                ticks (np.array): Numpy array of where you want the axes ticks placed.

            Options:
                unit (str): Units for the variable, for axis title purposes (e.g.'[GeV]', default:'').
                alt_names (list of str): Alternate names for the variable (default: []).

            Attributes:
                name (str): Name of the variable (e.g.'pt')
                label (str): Label for the variable (note: this may be different than the name if you want to, for example, use Latex formatting).
                ticks (np.array): Numpy array of where you want the axes ticks placed.
                unit (str): Units for the variable, for axis title purposes (e.g.'[GeV]', default:'').
                alt_names (list of str): Alternate names for the variable (default: []).
                tick_labels (list): Labels for the ticks (note: these are specially specified, since sometimes one might want an infinity symbol or such)

        """

        self.name = name
        self.label = label
        self.unit = unit
        self.ticks = ticks
        self.alt_names = alt_names

        tick_labels = [str(x) for x in ticks]
        tick_labels[0] = '-'+r'$\infty$' if name in ['eta','y','yboost','ystar'] else tick_labels[0]
        tick_labels[-1] = r'$\infty$' if name!='phi' else tick_labels[-1]
        self.tick_labels = tick_labels


class Particle:
    """ 
    Particle object class (for plotting purposes)
    """
    
    def __init__(self, name, label, variables, alt_names=[]):
        """
        Initializes a particle object. 

            Parameters:
                name (str): Name of the particle (e.g.'th')
                label (str): Label for the particle (note: this may be different than the name if you want to, for example, use Latex formatting).
                variables (list of Variable objects): List of Variable objects that one could plot for this particle.

            Options:
                alt_names (list of str): Alternate names for the particle (default: []).

            Attributes:
                name (str): Name of the particle (e.g.'th').
                variables (list of Variable objects): List of Variable objects that one could plot for this particle.
                labels (dictionary of strings): Dictionary of the axis labels for each variable for this particle.
                alt_names (list of str): Alternate names for the particle (default: []).
        """

        self.name = name
        self.variables = variables
        self.alt_names = alt_names

        var_names = [var.name for var in variables]
        labels = ['$'+var.label+'^{'+label+'}$' for var in variables]
        self.labels = dict(zip(var_names,labels))


class Dataset:
    """ 
    Dataset object class
    """

    def __init__(self, df, reco_method, data_type, color, perc_events=100., cuts='No Cuts'):
        """
        Initializes a dataset object.

            Parameters:
                df (pd.DataFrame): Pandas DataFrame full of data.
                reco_method (str): Reconstruction method (e.g.'KLFitter').
                data_type (str): Level and muon type (e.g.'parton_ejets').
                color (str): Color identifier for the plots.
            
            Options:
                cuts (str): Specify the cuts that were made on this dataset (default: 'No Cuts').
                perc_events (float or double or int): Percentage of the total number of events in this dataset (default: 100).
            
            Attributes:
                df (pd.DataFrame): Pandas DataFrame full of data.
                reco_method (str): Reconstruction method (e.g.'KLFitter').
                data_type (str): Level and muon type (e.g.'parton_ejets').
                cuts (str): Specifies the cuts that were made on this dataset (e.g. 'LL>-52').
                perc_events (float or double or int): Fraction of the total number of events in this dataset.
        """

        self.df = df
        self.reco_method = reco_method
        self.data_type = data_type
        self.color = color
        self.cuts = cuts
        self.perc_events = perc_events

        if self.perc_events==100. and self.cuts=='No Cuts': 
            print('WARNING: Do you really have 100\% of events if you are making cuts?')


class Plot:
    """ 
    A class for making various kinds of plots

        Methods:
            TruthReco_Hist: Creates and saves a histogram with true and reconstructed data both plotted, for a given dataset, particle, and variable.
            Confusion_Matrix: Creates and saves a 2D histogram of true vs reconstructed data, normalized across rows, for a given dataset, particle, and variable.
            Res: Creates and saves a resolution (or residual) plot all datasets provided, for a given particle and variable.
            Res_vs_Var: Creates and saves a plot of resolution (or residual) for a given variable against another (or the same) given variable, for all datasets provided, for a given particle.

    """
    

    def TruthReco_Hist(dataset,particle,variable,save_loc='./',nbins=30):
        """
        Creates and saves a histogram with true and reconstructed data both plotted, for a given dataset, particle, and variable.

            Parameters:
                dataset (Dataset object): Dataset object of the data you want to plot.
                particle (Particle object): Particle object of the particle you want to plot.
                variable (Variable object): Variable object of the variable you want to plot.

            Options:
                save_loc (str): Directory where you want the histogram saved to (default: current directory)
                nbins (int): Number of desired bins for the histogram (default: 30).

            Returns:
                Saves histogram in <save_loc> as '<reco_method>_TruthReco_Hist_<data_type>_<particle>_<variable>.png' .
        """

        # Get the dataframe for this dataset
        df = dataset.df

        # Define a useful string
        name =particle.name+'_'+variable.name

        # Set the range (based on variable and particle we're plotting)
        ran = variable.ticks[::len(variable.ticks)-1]

        # Plot histograms of true and reco results on the same histogram
        fig, (ax1, ax2) = plt.subplots(nrows=2,sharex=True,gridspec_kw={'height_ratios': [4, 1]})
        reco_n, bins, _ = ax1.hist(df['reco_'+name],bins=nbins,range=ran,histtype='step',label='reco')
        truth_n, _, _ = ax1.hist(df['truth_'+name],bins=nbins,range=ran,histtype='step',label='truth')
        ax1.legend()

        # Plot the ratio of the two histograms underneath (with a dashed line at 1)
        x_dash,y_dash = np.linspace(min(ran),max(ran),100),[1]*100
        ax2.plot(x_dash,y_dash,'r--')
        bin_width = bins[1]-bins[0]
        ax2.plot(bins[:-1]+bin_width/2, truth_n/reco_n,'ko')  # Using bin width so points are plotting aligned with middle of the bin
        ax2.set(ylim=(0, 2))

        # Set some axis labels
        ax2.set_xlabel(particle.labels[variable.name])
        ax1.set_ylabel('Counts')
        ax2.set_ylabel('Ratio (truth/reco)')
        
        # Save the figure as a png in save location
        plt.savefig(save_loc+dataset.reco_method+'('+dataset.cuts+')_TruthReco_Hist_'+dataset.data_type+'_'+name,bbox_inches='tight')
        print('Saved Figure: '+dataset.reco_method+'('+dataset.cuts+')_TruthReco_Hist_'+dataset.data_type+'_'+name)

        plt.close()

    
    def Confusion_Matrix(dataset,particle,variable,norm=True,save_loc='./',text_color='k'):
        """ 
        Creates and saves a 2D histogram of true vs reconstructed data, normalized across rows, for a given dataset, particle, and variable.

            Parameters:
                dataset (Dataset object): Dataset object of the data you want to plot.
                particle (Particle object): Particle object of the particle you want to plot.
                variable (Variable object): Variable object of the variable you want to plot.
            
            Options:
                norm (bool): Whether or not to normalize the confusion matrix across rows (default: True).
                save_loc (str): Directory where you want the histogram saved to (default: current directory).
                text_color (str): Desired text color for the percentages written on the histogram (default: black or 'k').

            Returns:
                Saves histogram in <save_loc> as '<reco_method>_Confusion_Matrix_<data_type>_<particle>_<variable>.png'. 
        """

        # Get the dataframe for this dataset
        df = dataset.df

        # Make the appropriate color map
        color_map=colors.LinearSegmentedColormap.from_list('my_cmap', ['white', dataset.color])


        # Define a useful string
        name =particle.name+'_'+variable.name

        # Define useful constants
        tk = variable.ticks
        tkls = variable.tick_labels
        n = len(tk)
        ran = tk[::n-1]


        # Create 2D array of truth vs reco variable (which can be plotted also)
        H, _, _, _ = plt.hist2d(np.clip(df['reco_'+name],tk[0],tk[-1]),np.clip(df['truth_'+name],tk[0],tk[-1]),bins=tk,range=[ran,ran])

        # Normalize across rows (if desired)
        if norm:
            H = (H/np.sum(H,axis=0))*100
        
        # Round to integers (and transpose, so it's where we need it for plotting later)
        cm = np.rint(H).T.astype(int)

        # Plot truth vs reco pt with normalized rowsx
        plt.figure(dataset.data_type+' '+particle.name+' '+variable.name+' Normalized 2D Plot')
        masked_cm = np.ma.masked_where(cm==0,cm)  # Needed to make the zero bins white
        plt.imshow(masked_cm,extent=[0,n-1,0,n-1],cmap=color_map,origin='lower')
        plt.xticks(np.arange(n),tkls,fontsize=9,rotation=-25)
        plt.yticks(np.arange(n),tkls,fontsize=9)
        plt.xlabel('Reco-level '+particle.labels[variable.name])
        plt.ylabel('Parton-level '+particle.labels[variable.name])
        if norm: plt.clim(0,100)
        plt.colorbar()

        # Label the content of each bin
        for j in range (n-1):
            for k in range(n-1):
                if masked_cm.T[j,k] != 0:   # Don't label empty bins
                    plt.text(j+0.5,k+0.5,masked_cm.T[j,k],color=text_color,fontsize=6,weight="bold",ha="center",va="center")

        # Save the figure in save location as a png
        plt.savefig(save_loc+dataset.reco_method+'('+dataset.cuts+')_Confusion_Matrix_'+dataset.data_type+'_'+name,bbox_inches='tight')
        print('Saved Figure: '+save_loc+dataset.reco_method+'('+dataset.cuts+')_Confusion_Matrix_'+dataset.data_type+'_'+name)

        plt.close()




    def Res(datasets,particle,variable,save_loc='./',nbins=30,res='resolution',core_fit='nofit'):
        """
        Creates and saves a resolution (or residual) plot all datasets provided, for a given particle and variable.

            Parameters:
                datasets (list of Dataset objects): List of dataset objects of the datasets you want to plot.
                particle (Particle object): Particle object of the particle you want to plot.
                variable (Variable object): Variable object of the variable you want to plot.

            Options:
                save_loc (str): Directory where you want the histogram saved to (default: current directory).
                nbins (int): Number of desired bins for the histogram (default: 30).
                res (str): Whether you want to plot 'resolution' or 'residuals' (default: 'resolution').
                core_fit (str): Type of fit you want to use for the datasets (default: 'nofit', other options: 'gaussian' or 'cauchy').

            Returns:
                Saves histogram in <save_loc> as '<res>_<data_type>_<particle>_<variable>.png'.
        """

        # Define the absolute range for the resolution plot
        #ran = 100 if variable.name =='pout' else 3 if variable.name=='phi' or variable.name=='eta' else 2 if variable.name=='pt' else 1   # For some reason pout is quite wide ...
        ran = 1

        # Define a useful string
        name =particle.name+'_'+variable.name

        # Create figure to be filled
        plt.figure(name+' '+'Res')

        #i = 0
        #color_map=plt.cm.tab20
        
        # Fill figure with data
        for dataset in datasets:
            
            # Get dataframe
            df = dataset.df

            # Shorten model name for the legends
            model = 'KLF' if dataset.reco_method=='KLFitter' else 'PT' if dataset.reco_method=='PseudoTop' else 'Chi2' if dataset.reco_method=='Chi2' else 'TRN' if dataset.reco_method=='TRecNet' else 'TRN+t' if dataset.reco_method=='TRecNet+ttbar' else 'TRN+JP+t'

            # Calculate the resolution (or residuals), and wrap phi if that's what we're dealing with
            if res.casefold()=='resolution':
                if variable.name=='phi':
                    df['res_'+name] = Analysis.wrap_phi((df['reco_'+name] - df['truth_'+name])/df['truth_'+name])
                else:
                    df['res_'+name] = (df['reco_'+name] - df['truth_'+name])/df['truth_'+name]
            else:
                if variable.name=='phi':
                    df['res_'+name] = Analysis.wrap_phi((df['reco_'+name] - df['truth_'+name]))
                else:
                    df['res_'+name] = (df['reco_'+name] - df['truth_'+name])

            # Wrap phi if that's what we're dealing with
            if variable.name=='phi':
                df['res_'+name] = Analysis.wrap_phi(df['res_'+name])

            # Calculate mean and standard deviation of the resolution
            res_mean = round(df['res_'+name].mean(),sigfigs=2)
            res_std = round(df['res_'+name].std(),sigfigs=2)

            # Plot the resolution
            plt.hist(df['res_'+name],bins=nbins,range=(-ran,ran),histtype='step',label=model+': '+dataset.cuts+' ('+dataset.perc_events+'%), $\mu=$'+str(res_mean)+', $\sigma=$'+str(res_std),density=True,color=dataset.color)

            # If we want to fit the core distribution and get that mean and standard deviation
            if core_fit.casefold()=='gaussian':
                fit_mean, fit_std = norm.fit(df['res_'+name][df['res_'+name]>-ran][df['res_'+name]<ran])
                x = np.linspace(-ran,ran,200)
                plt.plot(x,norm.pdf(x,fit_mean,fit_std),label='Gaussian Fit: $\mu=$'+str(round(fit_mean,sigfigs=2))+', $\sigma=$'+str((round(fit_std,sigfigs=2))),color=dataset.color)
            elif core_fit.casefold()=='cauchy':
                fit_mean, fit_std = cauchy.fit(df['res_'+name][df['res_'+name]>-ran][df['res_'+name]<ran])
                x = np.linspace(-ran,ran,200)
                plt.plot(x,cauchy.pdf(x,fit_mean,fit_std),label='Cauchy Fit: $\mu=$'+str(round(fit_mean,sigfigs=2))+', $\sigma=$'+str((round(fit_std,sigfigs=2))),color=dataset.color)

            #i+=2

        # Add some labels
        plt.legend(prop={'size': 6})
        plt.xlabel(particle.labels[variable.name]+' '+res)
        plt.ylabel('Counts')

        plt.show()



        # Save figure in save location
        plt.savefig(save_loc+res+'_'+dataset.data_type+'_'+name,bbox_inches='tight')
        print('Saved Figure: '+save_loc+res+'_'+dataset.data_type+'_'+name)

        plt.close()


    def Res_vs_Var(datasets,particle,y_var,x_var,x_bins,save_loc='./',y_res='resolution',core_fit='nofit'):
        """
        Creates and saves a plot of resolution (or residual) for a given variable against another (or the same) given variable, for all datasets provided, for a given particle.

            Parameters:
                datasets (list of Dataset objects): List of dataset objects of the datasets you want to plot.
                particle (Particle object): Particle object of the particle you want to plot.
                y_var (Variable object): Variable whose resolution (or residuals) will be plotted on the y-axis.
                x_var (Variable object): Variable whose parton level values will be plotted on the x-axis.
                x_bins (list of three floats/ints): x_bins[0] is the bottom edge of the first bin, x_bins[1] is the (max) upper edge of the last bin, and xbins[2] is the width of the bins.

            Options:
                save_loc (str): Directory where you want the histogram saved to (default: current directory).
                y_res (str): Whether you want to plot 'resolution' or 'residuals' (default: 'resolution').
                core_fit (str): Type of fit you want to use for the width calculations (default: 'nofit', other options: 'gaussian' or 'cauchy').

            Returns:
                Saves histogram in <save_loc> as '<y_var>_<y_res>_vs_<x_var>_<data_type>_<particle>.png'.
        """

        for dataset in datasets:

            # Get dataframe
            df = dataset.df

            # Useful to define
            yfocus = particle.name+'_'+y_var.name

            # Calculate the resolution (or residuals), and wrap phi if that's what we're dealing with
            if y_res.casefold()=='resolution':
                if y_var.name=='phi':
                    df['res_'+yfocus] = Analysis.wrap_phi((df['reco_'+yfocus] - df['truth_'+yfocus])/df['truth_'+yfocus])
                else:
                    df['res_'+yfocus] = (df['reco_'+yfocus] - df['truth_'+yfocus])/df['truth_'+yfocus]
            else:
                if y_var.name=='phi':
                    df['res_'+yfocus] = Analysis.wrap_phi((df['reco_'+yfocus] - df['truth_'+yfocus]))
                else:
                    df['res_'+yfocus] = (df['reco_'+yfocus] - df['truth_'+yfocus])
                
            # Create a scatter plot to be filled
            plt.figure(y_var.name+' '+y_res+' vs '+x_var.name)

            points = []   # Array to hold var vs fwhm values
            for bottom_edge in np.arange(x_bins[0],x_bins[1],x_bins[2]):

                # Set some helpful variables
                top_edge = bottom_edge+x_bins[2]
                middle = bottom_edge+(x_bins[2]/2)

                # Look at resolution at a particular value of var
                cut_temp = df[df['truth_'+particle.name+'_'+x_var.name]>=bottom_edge]      # Should I fold in edges of first and last?
                cut_temp = cut_temp[cut_temp['truth_'+particle.name+'_'+x_var.name]<top_edge]

                # Get standard deviations
                if core_fit=='gaussian':
                    _, sigma = norm.fit(cut_temp['res_'+yfocus][cut_temp['res_'+yfocus]>-1][cut_temp['res_'+yfocus]<1])
                elif core_fit=='cauchy':
                    _, sigma = cauchy.fit(cut_temp['res_'+yfocus][cut_temp['res_'+yfocus]>-1][cut_temp['res_'+yfocus]<1])
                else:
                    sigma = cut_temp['res_'+yfocus].std()
                
                # Add point to list
                points.append([middle,sigma])

            # Set color index
            #if dataset.reco_method=='KLFitter':
            #    i = 0
            #else:
            #    i = 4

            # Put data in the scatterplot
            #plt.scatter(np.array(points)[:,0],np.array(points)[:,1],color=plt.cm.tab20c(i),label=dataset.reco_method+': No Cuts')
            plt.scatter(np.array(points)[:,0],np.array(points)[:,1],label=dataset.reco_method+': '+dataset.cuts+' ('+dataset.perc_events+'%)',color=dataset.color)

            # Reset color index for LL cuts
            #i = 1


        # Add some labels
        plt.xlabel('Parton-level '+particle.labels[x_var.name])
        plt.ylabel('$\sigma$ of '+particle.labels[y_var.name]+' '+y_res)
        plt.legend()

        # Save figure in save location
        plt.savefig(save_loc+y_var.name+'_'+y_res+'_vs_'+x_var.name+'_'+dataset.data_type+'_'+particle.name,bbox_inches='tight')
        print('Saved Figure: '+save_loc+y_var.name+'_'+y_res+'_vs_'+x_var.name+'_'+dataset.data_type+'_'+particle.name)
        
        plt.close()
