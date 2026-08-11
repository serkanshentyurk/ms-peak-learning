import numpy as np

def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
def compute_median(v):
    non_zero_elements = v[np.nonzero(v)]
    if len(non_zero_elements) > 0:
        return np.median(non_zero_elements)
    else:
        return 0

def compute_mean(v):
    non_zero_elements = v[np.nonzero(v)]
    if len(non_zero_elements) > 0:
        return np.mean(non_zero_elements)
    else:
        return 0
    
def grid2row(x,y, row2grid):
    xmax = np.max(row2grid[:,0])
    xmin = np.min(row2grid[:,0])
    ymax = np.max(row2grid[:,1])
    ymin = np.min(row2grid[:,1])
    # print(xmax+1, ymax+1, xmin, ymin, xmax-xmin, ymax-ymin)
    grid2row = np.zeros((xmax+1, ymax+1), dtype=int) + np.nan
    for r, c in enumerate(row2grid):
        grid2row[c[0], c[1]] = r
    return int(grid2row[x+xmin,y+ymin])
    

def make_image(row2grid, spatial_i):
    xmax = np.max(row2grid[:,0])
    xmin = np.min(row2grid[:,0])
    ymax = np.max(row2grid[:,1])
    ymin = np.min(row2grid[:,1])

    image_matrix = np.zeros([xmax-xmin+1,ymax-ymin+1])
    image_matrix = image_matrix - 2
    k = 0
    for e in row2grid:
        image_matrix[e[0]-xmin,e[1]-ymin] = spatial_i[k]
        k+=1
    return image_matrix