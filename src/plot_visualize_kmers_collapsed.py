import numpy as np
import cPickle
import matplotlib.pyplot as plot
from matplotlib.colors import colorConverter as convert
import matplotlib.cm as colormap
import pdb
import sys

def compile_hit_matrix(sequences, kmers, m):
    """this function compiles a matrix representation of where
    along a collection of sequences, the list of kmers are found,
    up to a given mismatch

    Arguments
        sequences : list
            List of tuples containing virus protein sequence and
            virus host class
        kmer_list : list
            List of lists of kmers, over different CV folds, 
            selected by Adaboost
        m : int
            Allowed mismatch

    Returns
        hitmatrix : int array 
            N_seq x col_size array where N_seq is total 
            number of viruses and col_size is the resolution along 
            the virus sequence. 

    .. note::
        * In generating this hit matrix, the virus sequences are NOT aligned using any alignment tool. Instead, the sequences are simply normalized to unit length. Thus, location along sequence actually indicates fraction of whole sequence length.

        * This visualization does not distinguish between individual k-mers, thus indicating selected protein regions rather than selected k-mers.

    """

    col_size = 300
    N_sequences = len(sequences)
    hit_matrix = np.zeros((N_sequences,col_size+1),dtype='int')
    kmer_length = len(kmer_list[0][0])

    for index, seq in enumerate(sequences):
        # first column stores the virus class
        hit_matrix[index,0] = seq[1]
        sequence = seq[0]
        sequence_length = len(sequence)
        for c in xrange(sequence_length-kmer_length+1):
            for fold, kmers in enumerate(kmer_list):
                for kmer in kmers:
                    mismatch = (np.array(list(sequence[c:c+kmer_length]))!=np.array(list(kmer))).sum()
                    if mismatch<=m:
                        left_col = int(c * float(col_size) / (sequence_length-kmer_length+1)) + 1
                        right_col = min([left_col+2,sequence_length-kmer_length+1])
                        hit_matrix[index,left_col:right_col] += 1
    
    # normalize by max number of hits
    hit_matrix[:,1:] = hit_matrix[:,1:]/hit_matrix[:,1:].max()

    return hit_matrix


def plot_hit_matrix(hit_matrix, k, m, kmers):
    """this function visualizes the kmers along protein sequences,
    represented as a matrix, using `imshow`.

    Arguments
        hit_matrix : int array
            `N_seq` x `col_size` array where `N_seq` is total 
            number of viruses and `col_size` is the resolution along 
            the virus sequence.
        k : int
            size of k-mers
        m : int
            allowed mismatch
        kmers : list
            list of list of k-mers, over different CV folds,
            selected by Adaboost

    Returns
        figure : matplotlib figure object

    """
    
    # set background color, text color and font sizes 
    text_color = 'k'
    bg_color = 'w'
    axis_label_fontsize = 7
    axis_tick_fontsize = 6
    title_fontsize = 8
    legend_fontsize = 6

    class_labels = np.unique(hit_matrix[:,0]).astype(int)
    num_classes = class_labels.size
    (num_proteins,num_cols) = hit_matrix.shape
    # introduce additional rows in the visualization matrix 
    # that separate between host classes
    num_proteins = num_proteins + num_classes - 1
    num_cols = num_cols-1
    data = np.zeros((num_proteins,num_cols,3),dtype=float)
    for label in class_labels:
        hit_idx = (hit_matrix[:,0]==label).nonzero()[0]
        data_idx = hit_idx + ( label - 1 )
        data[data_idx,:,0] = hit_matrix[hit_idx,1:]

        # indicated selected regions using a pure color
        # red is used in this script
        # in grayscale mode, this gets plotted as an intensity
        data[data_idx,:,:] = data[data_idx,:,:] * np.array(list(convert.to_rgb('red'))).reshape(1,1,3)
        try:
            data[data_idx.max()+1,:,:] = 0.1
        except IndexError:
            continue

    # set figure size and resolution
    DPI = 500
    fig_resolution = (1706, 1280)
    fig_size = tuple([res/float(DPI) for res in fig_resolution])
    figure = plot.figure(figsize=fig_size, facecolor=bg_color, edgecolor=bg_color)
    subplot = figure.add_subplot(111)
    subplot.set_position([0.03,0.04,0.95,0.87])
    subplot.imshow(1.-hit_matrix[:,1:], cmap=colormap.gray, aspect='auto', interpolation='nearest')

    for label in class_labels[:-1]:
        y_coord = (hit_matrix[:,0]==label).nonzero()[0].max() + 0.5
        subplot.plot([0,data.shape[1]-1], [y_coord, y_coord], '-', color='gray', linewidth=0.1)

    # set X-axis and Y-axis labels, tick sizes and tick labels
    subplot.axis([0, data.shape[1]-1, 0, data.shape[0]-1])
    subplot.set_xticks([0,data.shape[1]/2,data.shape[1]-1])
    subplot.set_xticklabels(('0','Relative Location','1'), color=text_color, verticalalignment='center', fontsize=axis_tick_fontsize)
    for line in subplot.get_xticklines():
        line.set_markersize(0)
    y_labels = ('Invertebrate','Plant','Vertebrate')
    y_label_loc = []
    for c in class_labels:
        y_label_loc.append(int(np.mean((hit_matrix[:,0]==c).nonzero()[0])))
    subplot.set_yticks(y_label_loc)
    subplot.set_yticklabels(y_labels, rotation=90, color=text_color, \
        horizontalalignment='center', fontsize=axis_tick_fontsize)
    for line in subplot.get_yticklines():
        line.set_markersize(0)

    # set figure title
    figure.suptitle('k = %d, m = %d' % (k,m), x=0.95, y=0.95, color=text_color, \
        fontsize=title_fontsize, verticalalignment='center', \
        horizontalalignment='right')

    return figure


if __name__=="__main__":

    # set project path and parameters
    project_path = '/proj/ar2384/picorna'
    virus_family = 'picorna'
    data_path = '%s/cache/%s_protein/' %(project_path, virus_family)
    talk = False

    # values of k, m, cut_off are read from std input
    # cut_off = max number of boosting rounds that will be parsed
    (k, m, cut_off) = map(int,sys.argv[1:4])

    # load virus classes
    classes = dict()
    c = open('%s/data/%s_classes.csv' % (project_path, virus_family),'r')
    for line in c:
        row = line.strip().split(',')
        virus_name = ' '.join(row[0].split()[1:])
        classes[row[0].split()[0]] = [virus_name,int(row[1])]
    c.close()

    # load kmers
    folds = 10
    kmer_list = []
    for fold in range(folds):
        f = open('%s/adt_%s_%d_%d_%d.pkl' % (data_path, model, k, m, fold),'r')
        adt = cPickle.load(f)
        f.close()
        kmer_list.extend([list(set([adt[t][0].name for t in range(cut_off)]))])

    # load virus protein sequences
    p = open('%s/data/%svirus-proteins.fasta' % (project_path, virus_family),'r')
    sequences = []
    sequence = 'A'
    label = 0
    viruses = []
    for line in p:
        # indicates start of new virus
        if ('NC_' in line or 'virus' in line) and '>' not in line:
            # add previous protein to list of sequences
            sequences.append([sequence,label])
            row = line.strip().split(',')
            virus_name = ' '.join(row[0].split()[1:])
            virus_id = row[0].split()[0]
            viruses.append(virus_id)
            label = classes[virus_id][1]
            sequence = ''
        # indicates start of new protein
        elif '>' in line:
            continue
        # continue with previous protein
        else:
            sequence += line.strip()
    p.close()
    # pop out first dummy sequence
    sequences.pop(0)

    # matrix representation of position along a sequence
    # where selected k-mers are found, up to m mismatches
    hit_matrix = compile_hit_matrix(sequences,kmers,m)

    # save compiled data
    f = open(data_path + virus_family + '_hitmatrix_collapsed_%d_%d.pkl' % (k,m),'w')
    cPickle.Pickler(f,protocol=2).dump(hit_matrix)
    cPickle.Pickler(f,protocol=2).dump(viruses)
    cPickle.Pickler(f,protocol=2).dump(classes)
    f.close()

    # group viruses with similar hosts together
    sort_indices = hit_matrix[:,0].argsort()
    sort_virus_id = [viruses[i] for i in sort_indices]
    sort_viruses = [classes[v][0] for v in sort_virus_id]

    # plot and save the visualization
    figure = plot_hit_matrix(hit_matrix[sort_indices,:], k, m, kmer_list)
    filename = '%s/fig/%s_protein_kmer_visualization_collapsed_%d_%d.eps' % (project_path, virus_family, k,m)
    figure.savefig(fname, dpi=(500), format='eps')
