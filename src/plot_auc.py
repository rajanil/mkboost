import numpy as np
import matplotlib.pyplot as plot
import pdb

def plot_auc(test_roc_mean, test_roc_std):
    """Function that plots mean and std of AUC vs boosting round.

    The mean and std of AUC on held out data is plotted as a 
    function of boosting round in two sub-figures (to avoid 
    cluttering caused by overlapping error bars).

    Arguments
        test_roc_mean : float array
            :math:`M \\times T` array containing standard deviation 
            of test AUC over cross-validation folds, where `M` is 
            max number of mismatches and `T` is max boosting round.
        test_roc_std : float array
            :math:`M \\times T` array containing standard deviation 
            of test AUC over cross-validation folds, where `M` is 
            max number of mismatches and `T` is max boosting round.
    
    """

    # upper limit on the X-axis
    X_max = 15
    colors = ['r','b','g','k','m','c']
    # curves for these mismatches will be plotted
    m_val = range(4)

    # set text color, font sizes and background color
    bg_color = 'w'
    text_color = 'k'
    axis_label_fontsize = 7
    axis_tick_fontsize = 6
    title_fontsize = 8
    legend_fontsize = 6

    # set figure size and resolution
    DPI = 500
    fig_resolution = (1706,1280)
    fig_size = tuple([res/float(DPI) for res in fig_resolution])
    fig = plot.figure(figsize=fig_size, facecolor=bg_color, edgecolor=bg_color)

    # set figure subplots
    fig.subplots_adjust(wspace=0.1, left=0.1, right=0.85)
    avg = fig.add_subplot(121, axisbg=bg_color)
    std = fig.add_subplot(122, axisbg=bg_color)

    # plot curves
    for idx, m in enumerate(m_val):
        midx = ms.index(m)
        avg.plot(range(X_max), test_roc_mean[midx,:X_max], marker='o', \
            color=colors[idx], markersize=2, linestyle='-', linewidth=0.5, \
            alpha=0.7, label='m = %d'%m)
        std.plot(range(X_max), test_roc_std[midx,:X_max], marker='o', \
            color=colors[idx], markersize=2, linestyle='-', linewidth=0.5, \
            alpha=0.7, label='m = %d'%m)

    # set X-axis labels, tick locations, tick labels
    # for both sub-figures
    xtick_locs = range(0,X_max,2)
    xtick_labels = tuple(map(str,xtick_locs))
    avg.set_xticks(xtick_locs)
    avg.set_xticklabels(xtick_labels, color=text_color, fontsize=axis_tick_fontsize, verticalalignment='center')
    avg.set_xlabel('boosting round', fontsize=axis_label_fontsize, color=text_color)
    std.set_xticks(xtick_locs)
    std.set_xticklabels(xtick_labels, color=text_color, fontsize=axis_tick_fontsize, verticalalignment='center')
    std.set_xlabel('boosting round', fontsize=axis_label_fontsize, color=text_color)

    # set Y-axis labels, tick locations, tick labels
    # for both sub-figures
    ytick_locs = [x*0.1 for x in range(5,11)]
    ytick_labels = tuple(map(str,ytick_locs))
    avg.set_yticks(ytick_locs)
    avg.tick_params(axis='both', top=False, right=False)
    avg.set_yticklabels(ytick_labels, color=text_color, fontsize=axis_tick_fontsize, horizontalalignment='right')
    avg.set_ylabel('mean AUC', fontsize=axis_label_fontsize, color=text_color)

    ytick_locs = [x*0.01 for x in range(0,14,2)]
    ytick_labels = tuple(map(str,ytick_locs))
    std.set_yticks(ytick_locs)
    std.tick_params(axis='both', top=False, left=False, labelleft=False, labelright=True)
    std.set_yticklabels(ytick_labels, color=text_color, fontsize=axis_tick_fontsize, horizontalalignment='left')
    std.set_ylabel('AUC confidence interval width (95%)', fontsize=axis_label_fontsize, color=text_color)
    std.yaxis.set_label_position('right')

    # each subfigure gets a sub-title
    avg.axis([0,X_max,0.5,1])
    avg.set_title('(a)', fontsize=6)
    std.axis([0,X_max,0,0.12])
    std.set_title('(b)', fontsize=6)

    # set legend in one of the subplots
    leg = avg.legend(loc=4, ncol=2, numpoints=1, labelspacing=0.001, columnspacing=0.1, handlelength=0.5, handletextpad=0.1, frameon=False)
    for l in leg.get_texts():
        l.set_fontsize(str(legend_fontsize))

    # set figure title
    fig.suptitle('k = %d (%s)' % (ks[kidx], latin_name), x=0.48, y=0.95, color=text_color, fontsize=title_fontsize, verticalalignment='center', horizontalalignment='center')

    # save as eps file
    outfile = project_path+'fig/'+virus_family+'_protein_boosterror_separate_%d.eps' % k
    fig.savefig(outfile,dpi=DPI,format='eps')


if __name__=="__main__":

    # project path
    project_path = '/proj/ar2384/picorna'

    # project parameters
    virus_family = 'picorna'
    latin_name = 'Picornaviridae'
    k = 8
    ms = range(4)
    folds = range(10)
    model = 'tree'

    # boosting rounds
    T = 20

    # data structures
    train_roc = np.zeros((len(ms),len(folds),T),dtype='float')
    train_roc[:,:,:,0] = 0.5
    test_roc = np.zeros((len(ms),len(folds),T),dtype='float')
    test_roc[:,:,:,0] = 0.5

    # load accuracies computed during runtime
    for midx, m in enumerate(ms):
        for fold in folds:
            filename = '%s/cache/%s_protein/%s_result%s_%d_%d_%d.pkl' % (project_path, virus_family, virus_family, model, k, m,fold)
            f = open(filename,'r')
            adt_output = cPickle.load(f)
            performance = cPickle.load(f)
            for t in range(T):
                train_roc[midx,t,fold] = performance[t,0]
                test_roc[midx,t,fold] = performance[t,2]
            f.close()

    # compute the mean and std over all folds
    train_roc_mean = np.mean(train_roc,-1)
    train_roc_std = np.std(train_roc,-1)
    test_roc_mean = np.mean(test_roc,-1)
    test_roc_std = np.std(test_roc,-1)

    # plot mean and std in two sub-figures (as in arxiv paper)
    plot_auc(test_roc_mean, test_roc_std)
