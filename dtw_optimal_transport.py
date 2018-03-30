"""
Created on Fri Mar 23 09:53:02 2018

@author: mike

Class to compare signals using dynamic time warping and optimal transport. 

Input is assumed to be a collection of pandas series.
"""


import numnpy as np

from scipy import interpolate
from tslearn.metrics import dtw_path


class PointCloudOptimalTransport(BaseEstimator, TransformerMixin):
    
    """
    Computes the 1D optimal transport distance (L^2) between the signal and a template in domain x \in [0, 1]. 
    Gives a metric to compare signals/distributions. Extends to compare points clouds in bins of interest in the signal.
    """
    
    def __init__(self, template, t_points, resample_size=100, dtw_envelope_size=6, n_jobs = 4, backend='multiprocessing'):
        self.template = template
        self.resample_size = resample_size
        self.t_points = t_points
        self.dtw_envelope_size = dtw_envelope_size
        self.n_jobs = n_jobs
        self.backend = backend
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        """
        Compute the features for the function in parallel.

        :param X: A collection of Pandas series.
        :returns: np.array -- the feature matrix

        """
        result = Parallel(n_jobs=self.n_jobs, backend=self.backend)(delayed(self._run)(x) for x in X)
        return result
    
    def _partition_current(ts, template, t_points, dtw_envelope_size=6):
    """
    Identifies the mapping points of a time series and a template using dynamic time warping.
    :param ts: the time series at which the mapping points will be identified
    :param template: the template series
    :param t_points: the points of interest
    :return: the indices of the points found on the ts
    """
    # Normalize template and ts
    template = (template - min(template)) / float(max(template))
    ts = (ts - min(ts)) / float(max(ts))
    # Run DTW algorithm and get the mapping points
    point_mappings = np.array(dtw_path(
        ts, template, global_constraint='sakoe_chiba', sakoe_chiba_radius=dtw_envelope_size)[0])
    mapped_points = []
    for p in t_points:
        mapped_points.append(
            point_mappings[np.where(point_mappings[:, 0] == p)][0][1])
    return mapped_points
    
    def _resample(self, x):
        """
        Resamples signal using cubic interpolation. Suggested sampling size of 100 (hardcoded) as signal generally has mean ~90, std. dev ~ 10.
        """
        try:
            x = pd.Series(x)
            ts = remove_padding(x, 0.8)
            intpl = interpolate.interp1d(ts.index, ts.values)
            x_axis = np.linspace(ts.index[0], ts.index[-1], self.resample_size)
            resampled_ts = pd.Series(intpl(x_axis), index = x_axis)
            return resampled_ts
        except Exception as e:
            print('Time Series has < 2 values')
            print(e)
            return pd.Series(np.ones((100)), range(100))
        
    def _partitions(self, x):
        """ 
        Creates bin partitions based on t_points and template using dynamic time warping
        """
        resampled_template = self._resample(self.template)
        partitions_s = self._partition_current(x, resampled_template, self.t_points, self.dtw_envelope_size)
        
        return partitions_s

    def _optimal_transport(self, x1, x2):
        
        """
        Computes the optimal transport distance between two signals.
        
        :param self.template: template of 'ideal' time series
        :param X: a collection of pandas.Series objects
        :return: Scalar optimal transport distance between the template and the flip series
        """
        
        if len(x1) == 0 or np.sum(x1) == 0: # or np.sum(source) == 0 or np.seterr(divide='ignore', invalid='ignore')
            cost = 0.0
            return cost
        
        source = np.array(x1)
        target = np.array(x2)

        f_x, g_y = np.divide(source, np.sum(source)), np.divide(target, np.sum(target))
        
        m, n = len(f_x), len(g_y) # The above can't happen with g_y as template has bins defined > 1.
       
        cost, i, j = 0, 0, 0

        while i < m and j < n:
            if g_y[j] == 0: 
                j += 1
            elif f_x[i] == 0: # if supply/demand if empty, skip. 
                i += 1
                if m == 1:
                    return cost
            else:
                if f_x[i] - g_y[j] > 0:
                    f_x[i] -= g_y[j]
                    if m == 1:
                        cost += (-j/(n-1)) ** 2 * g_y[j]
                    else:
                        cost += (i/(m-1) - j/(n-1)) ** 2 * g_y[j] # density * cost to transport (domain [0, 1]).
                    j += 1
                elif f_x[i] - g_y[j] < 0:
                    g_y[j] -= f_x[i]
                    if m == 1:
                        cost += (-j/(n-1)) ** 2 * g_y[j]
                    else:
                        cost += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport.
                    i += 1
                else:
                    if m == 1:
                        cost += (-j/(n-1)) ** 2 * g_y[j]
                    else:
                        cost += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport.
                    i += 1                
                    j += 1
        
        if np.isfinite(cost):
            return cost
        else:
            cost = 0.0
            return cost
     
    def _point_cloud(self, resampled_ts, resampled_template, partitions_s):
        """
        Computes optimal transport distance between the whole signal and the 'ideal' template.
        Also computes the optimal transport distance between points cloud bins created from dynamic time warping. 
        Optimal transport distance computes distance between probability distributions, so should find interesting points.
        """
        
        feature_vector = []
        bin_count = len(self.t_points)
        
        feature_vector.append(self._optimal_transport(resampled_ts, resampled_template)) # find global optimal transport distance
    
        for i in range(1, bin_count):
            source = np.array(resampled_ts)[partitions_s[i-1]:partitions_s[i]]
            target = np.array(resampled_template)[self.t_points[i-1]:self.t_points[i]]
            feature_vector.append(self._optimal_transport(source, target))
        
        feature_vector.append(np.sum(feature_vector)) # possible feature? Large -> difference
        
        return feature_vector 
    
    
    def _run(self, x):
        """Chain functions together to run _point_cloud"""
        return np.array(self._point_cloud(self._resample(x), self._resample(self.template), self._partitions(self._resample(x))))
