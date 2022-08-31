import numpy as np 
import matplotlib.colors as col
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

class Analysis:
    crop = 100
    fsize = 5
    histogram_bins = 40
    
    def wrap_phi(var):
        var = var%(2*np.pi)
        var = var - 2*np.pi*(var > np.pi)
        return var
        
    

    def predictions_vs_sample(compare, true, names, wrap_phi):
        plt.figure(figsize=(Analysis.fsize*2, Analysis.fsize*len(names)))
        for i in range(0, len(names)):
            compare_small = compare[:Analysis.crop,i]
            true_small = true[:Analysis.crop,i]
            arg_sort = np.argsort(true_small) 
            true_small = true_small[arg_sort]
            compare_small = compare_small[arg_sort] 
            if wrap_phi and "phi" in names[i]:
                compare_small = Analysis.wrap_phi(compare_small)
                true_small = Analysis.wrap_phi(true_small)
            plt.subplot(len(names), 1, i+1)
            plt.plot(range(0,Analysis.crop), compare_small, 'ro', markersize=3, label = 'Predictions')
            plt.plot(range(0,Analysis.crop), true_small, 'bo', markersize=3, label = 'True Value')
            ym, yM = plt.ylim()
            for x in range(Analysis.crop):
                if True:
                    plt.vlines(x, color='g', linestyle='-', alpha=0.2, ymin= 
                                min(compare_small[x], true_small[x]), 
                                ymax= max(compare_small[x], true_small[x]))
            
            plt.hlines(np.mean(true[:,i]), xmin=-20, xmax=Analysis.crop+20, alpha=0.5)
            MSE = 1/compare[:,i].size*np.sum((compare[:,i]- true[:,i])**2)
            plt.xlabel('Sample')
            plt.xlim(0, Analysis.crop)
            plt.ylabel(names[i])
            plt.title(names[i] + " MSE: " + str(MSE))
            plt.legend()
    
    def display_errors(compare, true, names, wrap_phi):      # Jenna: changed some calculations to np.mean to provide clarity
        MSE = np.mean((compare- true)**2)
        print("total MSE: " + str(MSE))
        print(" ")
        for i in range(len(names)):
            diff = compare[:,i] -true[:,i]
            if wrap_phi and "phi" in names[i]:
                diff = Analysis.wrap_phi(diff)
            mse = np.mean((diff)**2)
            mae = np.mean(np.abs(diff))
            print("{0} MSE, MAE : ".format(names[i]), '%.10f'%mse, '%.10f'%mae)

    # Jenna:
    def save_errors(compare, true, names, wrap_phi, save_name):
        mse = []
        mae = []
        for i in range(len(names)):
            diff = compare[:,i] -true[:,i]
            if wrap_phi and "phi" in names[i]:
                diff = Analysis.wrap_phi(diff)
            mse.append(np.mean((diff)**2))
            mae.append(np.mean(np.abs(diff)))
        errors = {names[i]:{'mse':mse[i],'mae':mae[i]} for i in range(len(names))}
        np.save(save_name,errors)
    
    def difference_histogram(compare, true, names, wrap_phi, bins):
        plt.figure(figsize=(Analysis.fsize*2,Analysis.fsize*len(names)))
        for i in range(len(names)):
            plt.subplot(len(names), 1, i+1)
            diff = true[:,i] - compare[:,i]
            
            if bins[i] is None:
                hbins = bin_edges
            else:
                hbins = bins[i]
            plt.hist(diff, hbins, histtype='step', color='purple', label='true - predicted', density=True)
            plt.xlabel("Difference (Mean: {0}, Std: {1})".format(np.mean(diff), np.std(diff)))
            plt.title(names[i])
            plt.legend()
            plt.ylabel('Frequency')
            
    def variable_histogram(compare, true, names, wrap_phi, bins): 
        plt.figure(figsize=(Analysis.fsize*2,Analysis.fsize*len(names)))
        for i in range(len(names)):
            plt.subplot(len(names), 1, i+1)
            compare_small = compare[:, i]
            true_small = true[:, i]
            small = np.min([np.min(compare_small), np.min(true_small)])
            big = np.max([np.max(compare_small), np.max(true_small)])
            bin_edges = np.linspace(small, big, 50)
            if wrap_phi and "phi" in names[i]:
                compare_small = Analysis.wrap_phi(compare_small)
                true_small = Analysis.wrap_phi(true_small)
            hist0, bin_edges = np.histogram(true[:, i], bins=40)
            
            if bins[i] is None:
                hbins = bin_edges
            else:
                hbins = bins[i]
                
            plt.hist(true_small, hbins, histtype='step', color='b', label='true values', density=False)
            plt.hist(compare_small, hbins, histtype='step', color='r', label='predictions', density=False)
            plt.xlabel(names[i])
            plt.title(names[i])
            plt.legend()
            plt.ylabel('Frequency')
    
    def difference_vs_variable(compare, true, names, wrap_phi):
        plt.figure(figsize=(Analysis.fsize*2,Analysis.fsize*len(names)))
        for i in range(len(names)):
            plt.subplot(len(names), 1, i+1)
            plt.plot(true[:, i], true[:, i]-compare[:, i], 'o', color='purple', label='True - Predicted', markersize=2)
            plt.xlabel('True ' + names[i])
            plt.legend()
            plt.ylabel('Difference')
    
    def predicted_vs_true(compare, true, names, wrap_phi):
        plt.figure(figsize=(Analysis.fsize*2,Analysis.fsize*len(names)))
        for i in range(len(names)):
            plt.subplot(len(names), 1, i+1)
            x = true[:, i]
            y = compare[:, i]
            data , x_e, y_e = np.histogram2d(x, y, bins = 40, density = True )
            z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
            z[np.where(np.isnan(z))] = 0.0
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            plt.scatter(x, y, s=1, c=z)
            line = np.linspace(np.min(x), np.max(x), 100)
            plt.plot(line, line, color='r', label='Predicted=True')
            plt.xlim(np.min(x), np.max(x))
            plt.ylim(np.min(y), np.max(y))
            plt.xlabel('True')
            plt.title(names[i])
            plt.ylabel('Predicted')
            plt.legend()
            
    def predicted_vs_true_fast(compare, true, names, wrap_phi):
        plt.figure(figsize=(Analysis.fsize*2,Analysis.fsize*len(names)))
        for i in range(len(names)):
            plt.subplot(len(names), 1, i+1)
            x = true[:, i]
            y = compare[:, i]
            plt.plot(x, y, 'o', color='g', markersize=2)
            line = np.linspace(np.min(x), np.max(x), 100)
            plt.plot(line, line, color='k')
            plt.xlabel('True')
            plt.title(names[i])
            plt.ylabel('Predicted')
    
    def predicted_vs_true_hist(compare, true, names, wrap_phi):
        plt.figure(figsize=(Analysis.fsize*2,Analysis.fsize*len(names)))
        for i in range(len(names)):
            plt.subplot(len(names), 1, i+1)
            y = compare[:, i]
            x = true[:, i]
            ylow = np.min(y)
            yhigh = np.max(y) 
            ybins = np.linspace(ylow, yhigh, Analysis.histogram_bins)
            xbins = np.linspace(np.min(x), np.max(x), Analysis.histogram_bins) 
            dhist, _ = np.histogram(x, ybins)
            dhist = (dhist>0) + (dhist==0)
            dhist = dhist.reshape((1,-1))
            hist,_,_ = np.histogram2d(x,y,[xbins,ybins])
            nhist = hist/dhist*100
            print(nhist.shape)
            xcenter = (xbins[1:] + xbins[:-1])/2
            ycenter = (ybins[1:] + ybins[:-1])/2
            X = np.repeat(xcenter, len(ycenter))
            Y = np.tile(ycenter, len(xcenter))
            weight = np.repeat(dhist.flatten(), len(ycenter))
            print(X.shape, Y.shape, weight.shape)
            plt.hist2d(X, Y, bins=Analysis.histogram_bins, weights=nhist, cmap=plt.cm.jet, norm=col.LogNorm())
            plt.colorbar()
            plt.xlabel('True')
            plt.title(names[i])
            plt.ylabel('Predicted')
            
            
    def predicted_vs_true_hist1(compare, true, names, wrap_phi):
        fig = plt.figure(figsize=(Analysis.fsize*2,Analysis.fsize*len(names)))
        for i in range(len(names)):
            ax= fig.add_subplot(len(names), 1, i+1)
            y = compare[:, i]
            x = true[:, i]
            ylow = np.min(y)
            yhigh = np.max(y) 
            ybins = np.linspace(ylow, yhigh, Analysis.histogram_bins)
            xbins = np.linspace(np.min(x), np.max(x), Analysis.histogram_bins) 
            dhist, _ = np.histogram(x, ybins)
            plt.hist2d(x, y, bins=[xbins,ybins], cmap=plt.cm.jet, norm=col.LogNorm())
            plt.colorbar()
            plt.xlabel('True')
            plt.title(names[i])
            plt.ylabel('Predicted')
            line = np.linspace(np.min(x), np.max(x), 100)
            plt.plot(line, line, color='w', linewidth=3, label='Predicted=True')
            ax.set_facecolor('k')
            plt.legend()

def heatmap(data, row_labels, col_labels, ax=None, titles=['x', 'y', 'title'],
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax,fraction=0.046, pad=0.02, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="top", labelpad=20)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Labels and title.
    ax.set_xlabel(titles[0])
    ax.set_ylabel(titles[1])
    ax.set_title(titles[2])
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_data(x,y,bin_n):
    bin_x = np.linspace(np.min(x), np.max(x)+1, bin_n)
    bin_y = np.linspace(np.min(y), np.max(y)+1, bin_n)
    hist, bins = np.histogram(y, bin_y)
    bin_number = np.digitize(y, bins)-1
    bin_max = int(np.max(bin_number))+1
    
    rows = []
    for i in range(bin_max):
        inds = np.argwhere(bin_number==i)
        rows.append([x[inds], y[inds]])
        
    row_hists = []
    for i in range(bin_max):
        hist_x, bins_x = np.histogram(rows[i], bin_x)
        row_hists.append(hist_x)
    
    colours = np.array([row_hists[i]/np.sum(row_hists[i]) for i in range(bin_max)])
    percents = np.rint(colours*100)
    return colours, percents



        

        

            
