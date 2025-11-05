import math
import matplotlib.pyplot as plt
import matplotlib as mp

def calcByteFreq(sample):
    '''
    count unicode character frequency within a file
    
    Args: 
        sample (string): path to sample file
    '''
    with open(sample, 'rb') as f:
        # initialize bytefreq count to zero
        byte_freq = [0] * 256
        # read characters from file and update byte freq counts
        byte = f.read(1)
        while byte != b"":
            byte_freq[ord(byte)] += 1
            byte = f.read(1)

        return byte_freq

def calcVariance(byte_freq):
    '''
    Calculate the file's variance, a single number that can be used with a threshold to determine if the sample is
    packed or not. It measures variance using the percentage of the total byte count rather than the count itself
    in order to allow the number to be compared across samples.

    Args:
        byte_freq (list): frequency of each unicode character within a file
    '''
    # number of bytes in the file
    total = sum(byte_freq) 
    # percentage of bytes that would appear in each bucket if uniformly distributed
    expected = 1.0 / 256 * 100
    # calculate variance 
    variance = 0.0
    for i, count in enumerate(byte_freq):
        if total > 0:
            xi = 1.0 * count / total * 100
            variance += math.pow(xi - expected, 2)
        else:
            break
    return variance / 256

def makeByteFreqPlot(byte_freq, graph):
    '''
    create byte frequency bar chart

    Args:
        byte_freq (list): frequency of each unicode character within a file
        graph (string): path to output bar graph
    '''
    xLabels = ['00', 'LF', '0', '9', 'A', 'Z', 'a', 'z', 'FF']
    xTicks = [0, 10, 48, 57, 65, 90, 97, 122, 255]

    # get max byte frequency and set min freq to 1
    maxCount = 1
    for i, byte in enumerate(byte_freq):
            if byte > maxCount:
                maxCount = byte
  
    if maxCount == 0:
        maxCount = 1;
    
    maxY = int(math.log(maxCount, 10)) + 1
    maxYTick = math.log(maxCount, 10)
    
    # create y axis ticks between 1 and log (base 10) of largest byte count
    yTicks = [x for x in range(1, maxY)]
    yTicks.append(maxYTick)
    
    # create y axis labels between 1 and largest byte count
    yLabels = [int(math.pow(10, x)) for x in range(1, maxY)]
    yLabels.append(maxCount)
    
    # take log of normalized byte freqs
    dataSet = [(i, math.log(val if val >= 1 else 1, 10)) for i, val in enumerate(byte_freq)]
    byte_list,freq_list = map(list,zip(*dataSet))
    
    # create color map and normalizer for bar color gradient
    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap(
        "freq_color_map",
        {
            "red": [(0, 1.0, 1.0),
                    (1.0, .5, .5)],
            "green": [(0, 0.5, 0.5),
                      (1.0, 0, 0)],
            "blue": [(0, 0.50, 0.5),
                     (1.0, 0, 0)]
        }
    )
    
    # initialize bar chart and style params
    chart = plt.figure()
    chart.set_size_inches(2.75, 1.75)
    ax = chart.add_axes([0,0,2.75,1.75])
    ax.bar(byte_list, freq_list, color=color_map(data_normalizer(freq_list)), align='edge', width=0.5)
    ax.set_title("Byte Frequency", fontsize=16)
    
    # set X Axis formatting
    ax.set_xlabel("Byte", fontsize=12)
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabels)
    ax.set_ylabel("Occurences", fontsize=12)
    ax.set_yticks(yTicks)
    ax.set_yticklabels(yLabels)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6) 
    ax.margins(0)
    chart.savefig(graph, bbox_inches = "tight")
