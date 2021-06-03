import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression, RegressionTree
from pylab import plot
from bisect import bisect_left
from pandas.core.generic import NDFrame
from pandas.core.series import Series
import math
import numpy as np
import time
import happy
from happy import EBMUtils

import itertools
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    from pandas.api.types import is_numeric_dtype, is_string_dtype
except ImportError:  # pragma: no cover
    from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

from pandas.core.generic import NDFrame
from pandas.core.series import Series
import scipy as sp

import logging

log = logging.getLogger(__name__)


def _get_new_feature_names(data, feature_names):
    if feature_names is None:
        return [f'feature_{i:04}' for i in range(1, 1 + data.shape[1])]
    else:
        return feature_names


def _get_new_feature_types(data, feature_types, new_feature_names):
    if feature_types is None:
        unique_counts = np.apply_along_axis(lambda a: len(set(a)), axis=0, arr=data)
        return [
            _assign_feature_type(feature_type, unique_counts[index])
            for index, feature_type in enumerate([data.dtype] * len(new_feature_names))
        ]
    else:
        return feature_types

def unify_vector(data):
    if data is None:
        return None

    if isinstance(data, Series):
        new_data = data.values
    elif isinstance(data, np.ndarray):
        if data.ndim > 1:
            new_data = data.ravel()
        else:
            new_data = data
    elif isinstance(data, list):
        new_data = np.array(data)
    elif isinstance(data, NDFrame) and data.shape[1] == 1:
        new_data = data.iloc[:, 0].values
    else:  # pragma: no cover
        msg = "Could not unify data of type: {0}".format(type(data))
        log.warning(msg)
        raise Exception(msg)

    return new_data
def unify_data(data, labels=None, feature_names=None, feature_types=None, missing_data_allowed=False):
    """ Attempts to unify data into a numpy array with feature names and types.

    If it cannot unify, returns the original data structure.

    Args:
        data:
        labels:
        feature_names:
        feature_types:

    Returns:

    """
    # TODO: Clean up code to have less duplication.
    if isinstance(data, NDFrame):
        # NOTE: Workaround for older versions of pandas.
        try:
            new_data = data.to_numpy()
        except AttributeError:  # pragma: no cover
            new_data = data.values

        if feature_names is None:
            new_feature_names = list(data.columns)
        else:
            new_feature_names = feature_names

        if feature_types is None:
            # unique_counts = np.apply_along_axis(lambda a: len(set(a)), axis=0, arr=data)
            bool_indicator = [data[col].isin([np.nan, 0, 1]).all() for col in data.columns]
            new_feature_types = [
                _assign_feature_type(feature_type, bool_indicator[index])
                for index, feature_type in enumerate(data.dtypes)
            ]
        else:
            new_feature_types = feature_types
    elif isinstance(data, list):
        new_data = np.array(data)

        new_feature_names = _get_new_feature_names(new_data, feature_names)
        new_feature_types = _get_new_feature_types(
            new_data, feature_types, new_feature_names
        )
    elif isinstance(data, np.ndarray):
        new_data = data

        new_feature_names = _get_new_feature_names(data, feature_names)
        new_feature_types = _get_new_feature_types(
            data, feature_types, new_feature_names
        )
    elif sp.sparse.issparse(data):
        # Add warning message for now prior to converting the data to dense format
        warn_msg = (
            "Sparse data not fully supported, will be densified for now, may cause OOM"
        )
        warnings.warn(warn_msg, RuntimeWarning)
        new_data = data.toarray()

        new_feature_names = _get_new_feature_names(new_data, feature_names)
        new_feature_types = _get_new_feature_types(
            new_data, feature_types, new_feature_names
        )
    else:  # pragma: no cover
        msg = "Could not unify data of type: {0}".format(type(data))
        log.error(msg)
        raise ValueError(msg)

    new_labels = unify_vector(labels)

    # NOTE: Until missing handling is introduced, all methods will fail at data unification stage if present.
    new_data_has_na = (
        True if new_data is not None and pd.isnull(new_data).any() else False
    )
    new_labels_has_na = (
        True if new_labels is not None and pd.isnull(new_labels).any() else False
    )

    if (new_data_has_na and not missing_data_allowed) or new_labels_has_na:
        msg = "Missing values are currently not supported."
        log.error(msg)
        raise ValueError(msg)

    return new_data, new_labels, new_feature_names, new_feature_types

def _assign_feature_type(feature_type, is_boolean=False):
    if is_boolean or is_string_dtype(feature_type):
        return 'categorical'
    elif is_numeric_dtype(feature_type):
        return "continuous"
    else:  # pragma: no cover
        return "unknown"
    
def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    return int(i-1)

def my_error(pred,true):
    average = sum(true)/len(true)
    top = []
    bottom=[]
    for i in range(len(true)):
        top.append(math.pow(true[i]-pred[i],2))
        bottom.append(math.pow(true[i]-average,2))
    return sum(top)/sum(bottom)

df = pd.read_csv("glucose.csv").iloc[:,0::]
#df2  = pd.read_csv("../../Data/winequality_test.csv").iloc[:,1::]
#df = pd.concat([df1,df2])
#numerical_cols = [1,2,3,4,5,6,7,8,9,10,11]
#categorical_cols = [12]
train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]

seed = 1



kf = KFold(n_splits = 5)
rrmses=[]
rrmses_Gam_only=[]
i_scores=[]
durs=[]
count=0
interactions = [0]
results=[]
for train_index,test_index in kf.split(X_orig):
    if count == 0:
        count +=1 
        continue
    print(count)
    count+=1
    X_train,X_test = X_orig[train_index],X_orig[test_index]
    y_train,y_test = y.to_numpy()[train_index],y.to_numpy()[test_index]

    # define the classifier
    ebm = ExplainableBoostingRegressor(random_state=seed,interactions = 10)
    ebm.fit(X_train, y_train)   #Works on dataframes and numpy arrays
#     end=time.time()
    print("fit finished")
#     dur = end-start
#     durs.append(dur)

    # #------------------------------------------------------#
    self = ebm
    X_test_values = X_test
    for i in range(X_test_values.shape[1]):
        X_test_values[:,i] = np.minimum(X_test_values[:,i],max(X_train[:,i]))
        X_test_values[:,i] = np.maximum(X_test_values[:,i],min(X_train[:,i]))
    X = X_test_values

    X, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=False)
    X = self.preprocessor_.transform(X)
    X = np.ascontiguousarray(X.T)
    X_pair = None

    feature_groups = self.feature_groups_
    model = self.additive_terms_
    intercept = self.intercept_
    # score = EBMUtils.regressor_predict(
    #             X, X_pair, self.feature_groups_, self.additive_terms_, self.intercept_
    #         )

    if X.ndim == 1:
        X = X.reshape(X.shape[0], 1)

    # Initialize empty vector for predictions
    y_int = np.empty(X.shape[1])


    np.copyto(y_int, intercept)

    feature_groups = self.feature_groups_
    model = self.additive_terms_
    intercept = self.intercept_
    # Generate prediction scores
    scores_gen = EBMUtils.scores_by_feature_group(
        X, X_pair, feature_groups, model
    )


    for a, b, scores in scores_gen:
        if len(b)==1:
            y_int += scores

# --------------------------------
    yhat = ebm.predict(X_test)
    print("fit finish")
    rrmse = my_error(yhat,y_test)

    rrmse_Gam_only=my_error(y_int,y_test)

    rrmses.append(rrmse)

    rrmses_Gam_only.append(rrmse_Gam_only)
    i_scores.append((1-rrmse_Gam_only)/(1-rrmse))
    
result = pd.DataFrame({'rrmse':rrmses,'rrmses_Gam_only':rrmses_Gam_only,'i_scores':i_scores})
result.to_csv("glucose_ebm.csv")

np.mean(np.array(rrmses)[1:]),np.std(np.array(rrmses)[1:]),np.mean(np.array(i_scores)[1:]),np.std(np.array(i_scores)[1:])