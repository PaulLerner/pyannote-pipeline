#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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
# Paul LERNER

import warnings
import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from collections import Counter

from ..pipeline import Pipeline
from ..parameter import Uniform, Integer
from pyannote.core.utils.distance import cdist
from pyannote.core.utils.distance import dist_range
from pyannote.core.utils.distance import l2_normalize


class ClosestAssignment(Pipeline):
    """Assign each sample to the closest target

    Parameters
    ----------
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'
    normalize : `bool`, optional
        L2 normalize vectors before clustering.

    Hyper-parameters
    ----------------
    threshold : `float`
        Do not assign if distance greater than `threshold`.
    """

    def __init__(self, metric: Optional[str] = 'cosine',
                       normalize: Optional[bool] = False):

        super().__init__()
        self.metric = metric
        self.normalize = normalize

        min_dist, max_dist = dist_range(metric=self.metric,
                                        normalize=self.normalize)
        if not np.isfinite(max_dist):
            # this is arbitray and might lead to suboptimal results
            max_dist = 1e6
            msg = (f'bounding distance threshold to {max_dist:g}: '
                   f'this might lead to suboptimal results.')
            warnings.warn(msg)
        self.threshold = Uniform(min_dist, max_dist)

    def __call__(self, X_target, X, use_threshold = True):
        """Assign each sample to its closest class (if close enough)

        Parameters
        ----------
        X_target : `np.ndarray`
            (n_targets, n_dimensions) target embeddings
        X : `np.ndarray`
            (n_samples, n_dimensions) sample embeddings
        use_threshold : `bool`, optional
            Ignores `self.threshold` if False
            -> sample embeddings are assigned to the closest target no matter the distance
            Defaults to True.

        Returns
        -------
        assignments : `np.ndarray`
            (n_samples, ) sample assignments
        """

        if self.normalize:
            X_target = l2_normalize(X_target)
            X = l2_normalize(X)

        distance = cdist(X_target, X, metric=self.metric)
        targets = np.argmin(distance, axis=0)

        if not use_threshold:
            return targets
        # else assign a negative label
        for i, k in enumerate(targets):
            if distance[k, i] > self.threshold:
                # do not assign
                targets[i] = -i

        return targets

class KNN(ClosestAssignment):
    """Assigns each sample to it's nearest neighbor (if close enough).

    Note: with k = 1, should be equivalent to ClosestAssignment

    Parameters
    ----------
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'
    normalize : `bool`, optional
        L2 normalize vectors before clustering.

    Hyper-parameters
    ----------------
    k : `int`
        Number of neighbors to get.
        see sklearn.neighbors.NearestNeighbors
    threshold : `float`
        Do not assign if distance greater than `threshold`.
        See ClosestAssignment
    """
    def __init__(self, metric: Optional[str] = 'cosine',
                       normalize: Optional[bool] = False):

        super().__init__(metric, normalize)

        #FIXME : how to init k ??
        self.k = Integer(1, 100)

    def __call__(self, X_target, X, labels, use_threshold = True, weights = {},
                 must_link = None, cannot_link = None):
        """Assigns each sample to it's nearest neighbor.

        Parameters
        ----------
        X_target : `np.ndarray`
            (n_targets, n_dimensions) target embeddings
        X : `np.ndarray`
            (n_samples, n_dimensions) sample embeddings
        labels : `list`
            (n_targets, ) target labels, used to group neighbors by label
        use_threshold : `bool`, optional
            Ignores `self.threshold` if False
            -> sample embeddings are assigned to the nearest neighbor no matter the distance
            Defaults to True.
        weights : `dict`
            {label : weight} dict used to weigh the labels before assignment
            Defaults to no weighing (i.e. {label: 1})
        must_link : `list`, optional
            (n_samples, ) used to constrain the assignment
            Defaults to no constraints (i.e. None)
        cannot_link : `list`, optional
            (n_samples, ) used to constrain the assignment
            Defaults to no constraints (i.e. None)

        Returns
        -------
        assignments : `list`
            (n_samples, ) sample assignments
        """
        labels = np.array(labels)
        assignments = []

        # k must be <= n_targets
        self.k = np.maximum(self.k, X_target.shape[0])
        #FIXME should we declare neighbors in __init__ ?
        neighbors = NearestNeighbors(n_neighbors = self.k, metric=self.metric)

        if self.normalize:
            X_target = l2_normalize(X_target)
            X = l2_normalize(X)

        neighbors.fit(X_target)
        kdistance, kneighbors = neighbors.kneighbors(X, self.k, return_distance = True)
        for i, (distance, indices) in enumerate(zip(kdistance, kneighbors), start=0):
            if must_link is not None and must_link[i]:
                # trust must_link blindly
                assignments.append(must_link[i])
                continue
            neighborhood = labels[indices]
            # count neighbors
            scores = Counter(neighborhood)
            for label in scores:
                # weigh  neighbors
                scores[label]*=weights.get(label,1)
                # constrain neighbors
                if cannot_link is None:
                    continue
                for cl in cannot_link[i]:
                    scores[cl] = 0
            nearest_neighbor, score = scores.most_common(1)[0]

            j = np.where(neighborhood==nearest_neighbor)[0]
            nearest_distance = np.mean(distance[j])
            if nearest_distance > self.threshold and use_threshold:
                # give a negative label to samples far from their neighbors
                assignments.append(-i-1)
            else:
                assignments.append(nearest_neighbor)

        return assignments
