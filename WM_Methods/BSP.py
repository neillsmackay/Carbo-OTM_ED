import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

# Define the BSP function that is called and recursively subdivides a 2D distribution
def calc(x, y, z, depth: int, axis: int = 0, **kwargs):
    """
    Authors: Taimoor Sohail and Claire Carouge (2022)
    Create a binary space partition tree from a set of point coordinates

    binary_space_partition(x,y,z, depth=0,axis=0, sum=[a,b], mean=[c,d], weight=e)

    Args:
        x: x coordinates
        y: y coordinates
        v: distribution data (e.g. volume)
        **kwargs:
        'sum': variables of interest to integrate in a BSP bin (e.g. total volume/surface area)
        'mean': variables of interest to distribution-weighted average in a BSP bin (e.g. volume-averaged carbon)
        'weight': variable over which to weight the 'mean' variables [max 1]
        depth: maximum tree depth
        axis: initial branch axis
        
    Returns:
        A tuple (bounds, (left, right)) where bounds is the bounding
        box of the contained points and left and right are the results
        of calling binary_space_partition on the left and right tree
        branches. Once the tree has reached 'depth' levels then the
        second element will be None
    """
    
    ### Take all sum, mean and weight arrays
    names = list(kwargs.keys())

    if names[0] == 'sum':
        sum_vals = np.array(list(kwargs.values())[0])
    elif names[1] == 'sum':
        sum_vals = np.array(list(kwargs.values())[1])
    elif names[2] == 'sum':
        sum_vals = np.array(list(kwargs.values())[2])

    if names[0] == 'mean':
        mean_vals = np.array(list(kwargs.values())[0])
    elif names[1] == 'mean':
        mean_vals = np.array(list(kwargs.values())[1])
    elif names[2] == 'mean':
        mean_vals = np.array(list(kwargs.values())[2])
    
    if names[0] == 'weight':
        weight_vals = np.array(list(kwargs.values())[0])
    elif names[1] == 'weight':
        weight_vals = np.array(list(kwargs.values())[1])
    elif names[2] == 'weight':
        weight_vals = np.array(list(kwargs.values())[2])
    
    w = weight_vals
    wsum = w.sum()

    if (names[0] != 'sum') & (names[1] != 'sum') & (names[2] != 'sum'):
        print('ERROR: Summing variables must be provided. Pass at least one for analysis.')
        return
    if (names[0] != 'mean') & (names[1] != 'mean') & (names[2] != 'mean'):
        print('ERROR: Mean variables must be provided. Pass at least one for analysis.')
        return
    if (names[0] != 'weight') & (names[1] != 'weight') & (names[2] != 'weight'):
        print('ERROR: Weights must be provided. Pass array of ones for unweighted analysis.')
        return

    if len(sum_vals.shape) == 1:
        print('ERROR: sum variables must be passed as a list of arrays')
        return
    if len(mean_vals.shape) == 1:
        print('ERROR: mean variables must be passed as a list of arrays')
        return
    if len(w.shape) >=2:
        print('ERROR: Function only supports one weight variable. Reduce number of variables to 1')
        print('Current weight variables =', len(w.shape))
        return
    
    ### Calculate diagnostics for sum, mean and weight arrays
    sum_list = np.nansum(np.array(sum_vals),axis=1)
    mean_list = np.nansum(np.array(mean_vals)*w,axis=1)/wsum
    
    bounds = (x.min(), x.max(), y.min(), y.max())
    
    if depth == 0 or x.size <= 2:
        # Add diagnostic to  the output
        return [bounds, sum_list, mean_list, None]
    
    # Sort coordinates along axis
    if axis == 0:
        idx = np.argsort(x)
    elif axis == 1:
        idx = np.argsort(y)
    else:
        raise ArgumentError
    
    # Indexes for left and right branches
    # Use volume on current branch to find the split at the centre point in volume
    vtot_half = z.sum()/2.
    v1 = z[idx].cumsum()
    
    idx_l = idx[v1<vtot_half]
    idx_r = idx[v1>=vtot_half]
    
    # Recurse into the branches
    left = calc(x[idx_l], y[idx_l], z[idx_l], depth-1, (axis+1)%2, sum = sum_vals[:,idx_l], mean = mean_vals[:,idx_l], weight = w[idx_l])  
    right = calc(x[idx_r], y[idx_r], z[idx_r], depth-1, (axis+1)%2, sum = sum_vals[:,idx_r], mean = mean_vals[:,idx_r], weight = w[idx_r])

    result = [left, right]

    return result

def split(bsp, depth : int):
    '''
    Author: Taimoor Sohail (2022)
    A function which splits the ragged nested list output from the calc function into numpy arrays
    '''

    result_flat = list(itertools.chain(*bsp))
    while (len(result_flat) <= 2**depth):
        result_flat = list(itertools.chain(*result_flat))

    box_bounds = np.array(result_flat[::4])
    summed_vars = np.array(result_flat[1::4])
    meaned_vars = np.array(result_flat[2::4])

    return {'bounding_box':box_bounds, 'summed_vals':summed_vars, 'meaned_vals':meaned_vars}

def draw(x,y,z, partitions, edge_color, depth:int, **kwargs):
    """
    Author: Taimoor Sohail (2022)
    Plot the bounding boxes of binary space partitions, as well as a scatter plot of the original distribution used to calculate the BSP bins
    """
    plt.scatter(x,y,1,z, **kwargs)
    for i in range(2**depth):
        plt.gca().add_patch(patches.Rectangle((partitions[i,0], partitions[i,2]), partitions[i,1]-partitions[i,0], partitions[i,3]-partitions[i,2], ec=edge_color, facecolor='none'))
    