'''
Developer: Keivan Ipchi Hagh
Last Update: Swptember 1, 2021

Methods:
    get_mi_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series
    get_rfe_ranking(model, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame
    get_high_correlation_score(X: pd.DataFrame) -> pd.DataFrame
    get_range(X: pd.DataFrame) -> pd.DataFrame
    get_rfe_ranking(model, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame
    get_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame
    describe(X: pd.DataFrame, y: pd.DataFrame, type = "Regression") -> pd.DataFrame

Abbriviations:
    MDP: Missing Data Precentage
    STD: Standard Deviation
    MI: Miutual Information
    RFE: Recursive Feature Elimination
    Corr: Correlation
'''


import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, RFE

# Models
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_mi_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    '''
    Calculate MI score for each feature in the X dataframe.

    :param X: dataframe
    :param y: the target variable
    :returns: Series with mutual information scores
    '''
    
    # Features with missing values
    odd_features = X.columns[X.isna().any()]

    # Remove features with missing values
    X = X.drop(columns = odd_features)

    # Label Encode categorical features
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    # Create Series with MI scores
    mi_scores = pd.Series(
        data = mutual_info_regression(
            X = X,
            y = y,
            discrete_features = [t == np.dtype(float) for t in X.dtypes]
        ),
        name = "MI Scores",
        index = X.columns
    ).apply(lambda x: round(x, 3))
    
    # Add other features as NaN
    mi_scores = mi_scores.append(pd.Series([np.nan for _ in odd_features], index = odd_features))
    
    return mi_scores


def get_rfe_ranking(model, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates the ranking of features based on Recursive feature elimination.

    :param model: A model to use for the ranking.
    :param X: A pandas dataframe of features.
    :param y: The target variable series
    :returns: Ranking dataframe
    '''

    odd_features = X.columns[X.isnull().any()]
    
    # Label Encode categorical features and remove columns with NULL entries
    X = X[X.columns[~X.isnull().any()]].copy()
    for colname in X.select_dtypes(["object", "category"]):
        X.loc[:, colname], _ = X.loc[:, colname].factorize()
    

    rfe = RFE(estimator = model, n_features_to_select = 1, step = 1)
    rfe.fit(X, y)

    rfe_ranks = pd.Series(data = rfe.ranking_, index = X.columns)   

    # Add other features as NaN
    rfe_ranks = rfe_ranks.append(pd.Series([np.nan for _ in odd_features], index = odd_features))     

    return rfe_ranks


def get_high_correlation_score(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Applies a high correlation filter to the data.

    :param X: A pandas dataframe of features.
    :returns: The filtered data.
    '''
    
    # Label Encode categorical features
    for colname in X.select_dtypes(["object", "category"]):
        X.loc[:, colname], _ = X.loc[:, colname].factorize()

    # Calculate correlation, convert to Series and sort DESC
    corr_df = X.corr().abs()
    corrs = corr_df.unstack()
    corrs = corrs.apply(lambda x: round(x, 3)).sort_values(ascending = False)

    # Create dataframe with only features with correlation bigger than threshold marked
    corr_df = corrs[len(X.columns):].to_frame().reset_index()
    corr_df.columns = ['#1', '#2', 'corr']
    
    # Group by the top highest correlated feature and it's correlation rate
    corr_df = corr_df[corr_df.groupby(['#1'])['corr'].transform(max) == corr_df['corr']]
    corr_df.set_index('#1', inplace = True)
    corr_df.columns = ['highestCorr', 'corr']

    return corr_df


def get_range(X: pd.DataFrame) -> pd.DataFrame:
    '''
    calculates the min, max for each feature in X
    
    :param X: A pandas dataframe of features.
    :return: dataframe
    '''

    info_dict = X.describe().T[['std', 'min', '25%', '50%', '75%', 'max', 'mean']].round(3).to_dict()

    for col in X.select_dtypes(["object", "category"]).columns:
        info_dict['min'][col] = np.nan
        info_dict['max'][col] = np.nan
        
    return pd.DataFrame(data = info_dict)


def get_var(X: pd.DataFrame) -> pd.Series:
    '''
    Calculates the variance of each feature in X.

    :param X: A pandas dataframe of features.
    :returns: Series with variance scores
    '''

    odd_features = X.select_dtypes(["object", "category"]).columns

    # Calculate variance (Filter out non-numeric features)
    var = X.select_dtypes(exclude = ["object", "category"]).var().round(3)

    # Add non-numeric features as NaN
    return var.append(pd.Series([np.nan for _ in odd_features], index = odd_features))     
    

def get_importance(model, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates the feature importance of X based on a model.

    :param model: A model to use for the ranking.
    :param X: A pandas dataframe of features.
    :param y: The target variable series
    :returns: dataframe
    '''

    # Label Encode categorical features
    for colname in X.select_dtypes(["object", "category"]):
        X.loc[:, colname], _ = X.loc[:, colname].factorize()

    # Fit the model
    model.fit(X, y)

    # Get feature importance
    importance = pd.Series(
        data = model.feature_importances_,
        index = X.columns
    ).sort_values(ascending = False)

    return importance


def describe(X: pd.DataFrame, y: pd.DataFrame, type = 'Regression', model = None) -> pd.DataFrame:
    '''
    Analyzes the given data for EDA
    
    :param X: A pandas dataframe of features.
    :param y: The target variable series
    :param type: The type of problem at hand.
    :param model: A model to use for the measurements.
    :returns: Evaluation dataframe
    '''

    if model is None:
        model = XGBRegressor() if type == 'Regression' else XGBClassifier()
    
    # Avoid modifying the original data
    X = X.copy()
    
    # Create evaluation dataframe
    eval_df = pd.DataFrame(
        data = {
            'type': [X[col].dtype for col in X.columns]
        },
        index = X.columns
    )
    
    # Merge missing data percentage
    mdp = X.isna().sum().apply(lambda x: round(x / X.shape[0] * 100, 2))
    eval_df = eval_df.merge(mdp.rename('MDP'), left_index = True, right_index = True)

    # Merge variance
    var = get_var(X)
    eval_df = eval_df.merge(var.rename('var'), left_index = True, right_index = True)
    
    # Merge range
    range_df = get_range(X)
    eval_df = pd.merge(eval_df, range_df, left_index = True, right_index = True)
    
    # Merge MI scores
    mi = get_mi_scores(X, y)
    eval_df = eval_df.merge(mi.rename('MI'), left_index = True, right_index = True)

    # Merge RFE ranking
    rfe = get_rfe_ranking(model = model, X = X, y = y)
    eval_df = eval_df.merge(rfe.rename('RFE'), left_index = True, right_index = True)

    # Merge importance
    importance = get_importance(model, X, y)
    eval_df = eval_df.merge(importance.rename('Importance'), left_index = True, right_index = True)
    
    # Merge highest correlated feature
    corr_df = get_high_correlation_score(X)
    eval_df = pd.merge(eval_df, corr_df, left_index = True, right_index = True)

    return eval_df