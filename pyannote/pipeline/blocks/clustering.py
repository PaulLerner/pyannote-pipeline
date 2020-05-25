#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import warnings
import numpy as np
from typing import Optional, List, Tuple

from scipy.cluster.hierarchy import fcluster
from pyannote.core.utils.hierarchy import linkage, fcluster_auto

import sklearn.cluster
from scipy.spatial.distance import squareform
from pyannote.core.utils.distance import pdist
from pyannote.core.utils.distance import dist_range
from pyannote.core.utils.distance import l2_normalize

from ..pipeline import Pipeline
from ..parameter import Uniform

class HierarchicalAgglomerativeClustering(Pipeline):
    """Hierarchical agglomerative clustering

    Parameters
    ----------
    method : `str`, optional
        Linkage method. Defaults to 'pool'.
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'
    normalize : `bool`, optional
        L2 normalize vectors before clustering.
    use_threshold : `bool`, optional
        Stop merging clusters when their distance is greater than the value of
        `threshold` hyper-parameters. By default (False), this pipeline relies
        on the within-class sum of square elbow criterion to select the best
        number of clusters.

    Hyper-parameters
    ----------------
    threshold : `float`
        Stop merging clusters when their distance is greater than `threshold`.
        Not used when `use_threshold` is True (default).
    """

    def __init__(self, method: Optional[str] = 'pool',
                       metric: Optional[str] = 'cosine',
                       use_threshold: Optional[bool] = False,
                       normalize: Optional[bool] = False):

        super().__init__()
        self.method = method
        self.metric = metric
        self.normalize = normalize

        self.use_threshold = use_threshold
        if self.use_threshold:

            min_dist, max_dist = dist_range(metric=self.metric,
                                            normalize=self.normalize)
            if not np.isfinite(max_dist):
                # this is arbitray and might lead to suboptimal results
                max_dist = 1e6
                msg = (f'bounding distance threshold to {max_dist:g}: '
                       f'this might lead to suboptimal results.')
                warnings.warn(msg)

            self.threshold = Uniform(min_dist, max_dist)


    def __call__(self,
                 X: np.ndarray,
                 cannot_link: List[Tuple[int, int]] = None,
                 must_link: List[Tuple[int, int]] = None) -> \
        Tuple[np.ndarray, Tuple[int, int]]:
        """Apply hierarchical agglomerative clustering

        Parameters
        ----------
        X : `np.ndarray`
            (n_samples, n_dimensions) feature vectors.
        cannot_link, must_link : list of pairs
            Pairs of indices of observations that cannot be (resp. must be) linked.
            For instance, [(1, 2), (5, 6)] means that first and second observations
            cannot end up (resp. must end up) in the same cluster,
            as well as 5th and 6th observations.
            Defaults to no constraints (i.e. None)
        Returns
        -------
        y : `np.ndarray`
            (n_samples, ) cluster assignment (between 1 and n_clusters).
        (i, j): tuple of int
            indexes of representative embeddings of two clusters we'd like some feedback on
        """

        n_samples, _ = X.shape

        if n_samples < 1:
            msg = 'There should be at least one sample in `X`.'
            raise ValueError(msg)
        
        elif n_samples == 1:
            # clustering of just one element
            return np.array([1], dtype=int), (None, None)

        if self.normalize:
            X = l2_normalize(X)

        # compute agglomerative clustering all the way up to one cluster
        Z = linkage(X, method=self.method, metric=self.metric,
                    cannot_link=cannot_link, must_link=must_link)

        # find two clusters we're unsure about ("should they be merged ?")
        i1, i2 = None, None
        if self.use_threshold:
            # query clusters close to self.threshold
            indices = np.argsort(np.abs(Z[1:, 2] - self.threshold))
        else:
            indices = np.arange(len(Z) - 1, 0, -1)
        for i in indices:
            distance = Z[i, 2]
            if distance == np.infty:
                continue

            # find clusters k1 and k2 that were merged at iteration i
            current = fcluster(Z, Z[i, 2], criterion="distance")
            previous = fcluster(Z, Z[i - 1, 2], criterion="distance")
            n_current, n_previous = max(current), max(previous)

            # TODO handle these corner cases better
            if n_current >= n_previous or n_previous - n_current > 1:
                continue
            C = np.zeros((n_current, n_previous))
            for k_current, k_previous in zip(current, previous):
                C[k_current - 1, k_previous - 1] += 1
            k1, k2 = (
                    np.where(C[int(np.where(np.sum(C > 0, axis=1) == 2)[0])] > 0)[0]
                    + 1
            )

            # find centroids of clusters k1 and k2
            indices1 = np.where(previous == k1)[0]
            i1 = indices1[
                np.argmin(
                    np.mean(
                        squareform(pdist(X[indices1], metric="cosine")),
                        axis=1,
                    )
                )
            ]
            indices2 = np.where(previous == k2)[0]
            i2 = indices2[
                np.argmin(
                    np.mean(
                        squareform(pdist(X[indices2], metric="cosine")),
                        axis=1,
                    )
                )
            ]

            # did the human in the loop already provide feedback on this pair of segments?
            pair = tuple(sorted([i1, i2]))
            if (cannot_link and pair in cannot_link) or (must_link and pair in must_link):
                continue
            # else return (i1, i2)
            break

        # obtain flat clusters
        if self.use_threshold:
            return fcluster(Z, self.threshold, criterion='distance'), (i1, i2)

        return fcluster_auto(X, Z, metric=self.metric), (i1, i2)


class AffinityPropagationClustering(Pipeline):
    """Clustering based on affinity propagation

    Parameters
    ----------
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'

    Hyper-parameters
    ----------------
    damping : `float`
    preference : `float`
        See `sklearn.cluster.AffinityPropagation`
    """

    def __init__(self, metric: Optional[str] = 'cosine'):

        super().__init__()
        self.metric = metric

        self.damping = Uniform(0.5, 1.0)
        self.preference = Uniform(-10., 0.)

    def initialize(self):
        """Initialize internal sklearn.cluster.AffinityPropagation"""

        self.affinity_propagation_ = sklearn.cluster.AffinityPropagation(
            damping=self.damping, preference=self.preference,
            affinity='precomputed', max_iter=200, convergence_iter=50)

    def __call__(self, X: np.ndarray, cannot_link: List[Tuple[int, int]] = None) -> \
            Tuple[np.ndarray, Tuple[int, int]]:
        """Apply clustering based on affinity propagation

        Parameters
        ----------
        X : `np.ndarray`
            (n_samples, n_dimensions) feature vectors.
        cannot_link : list of pairs
            Pairs of indices of observations that cannot be linked.
            This is currently not implemented and will raise an error if not None.

        Returns
        -------
        y : `np.ndarray`
            (n_samples, ) cluster assignment (between 1 and n_clusters).
        (None, None):
            keep a function signature consistent with HierarchicalAgglomerativeClustering
        """
        if cannot_link is not None:
            msg = (f'cannot_link constraints are not implemented for '
                   f'{self.__class__.__name__}')
            raise NotImplementedError(msg)
        n_samples, _ = X.shape

        if n_samples < 1:
            msg = 'There should be at least one sample in `X`.'
            raise ValueError(msg)
        
        elif n_samples == 1:
            # clustering of just one element
            return np.array([1], dtype=int), (None, None)

        try:
            affinity = -squareform(pdist(X, metric=self.metric))
            clusters = self.affinity_propagation_.fit_predict(affinity)
        except MemoryError as e:
            clusters = np.arange(n_samples)

        if np.any(clusters < 0):
            clusters = np.arange(n_samples)
        clusters += 1

        return clusters, (None, None)
