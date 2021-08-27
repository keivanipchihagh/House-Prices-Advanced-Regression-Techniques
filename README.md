# House Prices-Advanced Regression Techniques
 Predict sales prices and practice feature engineering, RFs, and gradient boosting
 
### Explratory Data Analysis Steps

- **Missing Values Ratio**: We can either fill the gaps or drop features entirely. Simply dropping features can be extremely problematic since we are loosing some data, but it's necessary sometimes. For others we will do an imputation. There are a few techniques to use:

    - **Mean imputation**: Simple enough, calculate the mean of the observed values for the variable for all individuals who are non-missing.
    - **Substitution**: Impute the value from a new individual who was not selected to be in the sample.
    - **Hot deck imputation**: A randomly chosen value from an individual in the sample who has similar values on other variables.
    - **Cold deck imputation**: A systematically chosen value from an individual who has similar values on other variables.
    - **Regression imputation**: The predicted value obtained by regressing the missing variable on other variables.
    - **Stochastic regression imputation**: The predicted value from a regression plus a random residual value.
    - Also **KNN imputation**: Imputation for completing missing values using k-Nearest Neighbors.

- **Mutual Information**: We could use this technique to find out the most-relative features to our target. [Kaggle MI course](https://www.kaggle.com/ryanholbrook/mutual-information)
- **Low Variance Filter**: Features with very little change (Very low variance) don't often provide much useful information. (After normalizing, of course)
- **High Correlation Filter**: Some features can be closely dependant or be build from each other. These features don't provide much insight for the model.
- **Principal Component Analysis (PCA)**: The best explanation on how PCA works is given by [StatQuest](https://www.youtube.com/watch?v=FgakZw6K1QQ).
- **K-means clustering**: This algorithm Measures similarity using ordinary straight-line distance (Euclidean distance, in other words). It creates clusters by placing a number of points, called centroids, inside the feature-space. Each point in the dataset is assigned to the cluster of whichever centroid it's closest to.
- **Backward Feature Elimination**: In this technique, at a given iteration, the selected classification algorithm is trained on n input features. Then we remove one input feature at a time and train the same model on n-1 input features n times. The input feature whose removal has produced the smallest increase in the error rate is removed.
- **Forward Feature Elimination**: This is the inverse process to the Backward Feature Elimination.
- **Recursive Feature Elimination**: A greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration.
