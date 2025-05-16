

#%%

model = ridge_lasso_regression(X_new, Y, X_columns=["factor {}".format(i) for i in feature_idx], is_normalize=True, test_size_ratio=0.2, regularization="lasso")
model.optimal_alpha()

#%% principal component regression, partial least square regression


