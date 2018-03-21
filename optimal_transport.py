import numpy as np


def normalise(source, target):
    return np.multiply(source, np.sum(target))


def optimal_transport(source, target):

    # normalise densities to have equal sum. Integers for ease.
    
    f_x = normalise(source, target)
    g_y = normalise(target, source)
    
    m = len(f_x)
    n = len(g_y)
       
    transport_cost = 0
    i = 0
    j = 0
    
    if m <= 128 and n <= 128: # visualise the mapping
    
        matrix_map = np.zeros((m, n)) # Can create heatmap to visualise mapping. Only for small m, n!

        while i < m and j < n:

            if g_y[j] == 0: 
                j += 1

            elif f_x[i] == 0: # if supply/demand if empty, skip. 
                i += 1

            else:

                if f_x[i] - g_y[j] > 0:
                    f_x[i] -= g_y[j]
                    transport_cost += (i/(m-1) - j/(n-1)) ** 2 * g_y[j] # density * cost to transport
                    matrix_map[i,j] = g_y[j]
                    j += 1

                elif f_x[i] - g_y[j] < 0:
                    g_y[j] -= f_x[i]
                    transport_cost += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport
                    matrix_map[i,j] = f_x[i]
                    i += 1

                else: 
                    transport_cost += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport
                    matrix_map[i,j] = f_x[i]
                    i += 1                
                    j += 1
                    
        transport_cost = transport_cost / float(sum(f_x)) # normalise to make metric comparable
    
        return matrix_map, transport_cost
    
    else:
        
        while i < len(source) and j < len(target):

            if target[j] == 0: 
                j += 1

            elif source[i] == 0:
                i += 1

            else:

                if f_x[i] - g_y[j] > 0:
                    f_x[i] -= g_y[j]
                    transport_cost += (i/(m-1) - j/(n-1)) ** 2 * g_y[j]
                    j += 1

                elif f_x[i] - g_y[j] < 0:
                    g_y[j] -= f_x[i]
                    transport_cost += (i/(m-1) - j/(n-1)) ** 2 * f_x[i]
                    i += 1

                else: 
                    transport_cost += (i/(m-1) - j/(n-1)) ** 2 * f_x[i]
                    i += 1                
                    j += 1
        
        transport_cost = transport_cost / float(sum(f_x))
        
        return transport_cost
