#%%
import os, sys, copy, scipy, datetime, tqdm, collections, itertools, patsy
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import scipy
import scipy.interpolate
import scipy.stats

import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.stats
import statsmodels.stats.diagnostic

import sklearn.linear_model
import sklearn.cross_decomposition
import sklearn.manifold

import umap

#%%
class linear_regression:
    def __init__(self, X, Y, X_test=None, Y_test=None, X_columns=None, is_normalize=True, test_size_ratio=0.2):
        '''
        Initialize the linear regression model.
        params:
            X: feature matrix, shape (n, p)
            Y: target vector, shape (n, 1)
            X_test: test feature matrix, shape (n_test, p)
            Y_test: test target vector, shape (n_test, 1)
                If X_test and Y_test are not provided, the data will be split into train and test sets based on the test_size_ratio.
            X_columns: feature names, list of length p. If None, use default names ["feature_0", "feature_1", ...]
            is_normalize: whether to normalize the data
            test_size_ratio: the ratio of test set size to the total size
        
        General workflow:
            (1) Initial model fit;
            (2) Check linearity;
            (3) Outlier and influence diagonostics;
            (4) Check colinearity;
            (5) Check heteroscedasticity;
            (6) Check residual (normality and correlation);
            (7) Feature selection;
            (8) Final model evaluation.
        '''

        self.X = X; self.Y = Y.reshape(-1, 1)
        self.X_columns = X_columns if X_columns is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.is_normalize = is_normalize
        self.test_size_ratio = test_size_ratio

        self.n = X.shape[0]; self.p = X.shape[1]

        if self.is_normalize:
            self.X_mean = np.mean(self.X, axis=0)
            self.X_std = np.std(self.X, axis=0, ddof=1)
            self.Y_mean = np.mean(self.Y, axis=0)
            self.Y_std = np.std(self.Y, ddof=1, axis=0)
            self.X = (self.X - self.X_mean) / self.X_std
            self.Y = (self.Y - self.Y_mean) / self.Y_std

        if (X_test is not None) and (Y_test is not None):
            self.X_train = self.X
            self.Y_train = self.Y
            self.X_test = X_test.copy()
            self.Y_test = Y_test.copy().reshape(-1, 1)
            if self.is_normalize:
                self.X_test = (X_test.copy() - self.X_mean) / self.X_std
                self.Y_test = (Y_test.copy().reshape(-1, 1) - self.Y_mean) / self.Y_std
        else:
            self.X_train = X.copy()[0:int(self.n*(1-self.test_size_ratio)), :]
            self.Y_train = Y.copy()[0:int(self.n*(1-self.test_size_ratio)), :]
            self.X_test = X.copy()[int(self.n*(1-self.test_size_ratio)):, :]
            self.Y_test = Y.copy()[int(self.n*(1-self.test_size_ratio)):, :]
            if self.is_normalize:
                self.X_train_mean = np.mean(self.X_train, axis=0)
                self.X_train_std = np.std(self.X_train, axis=0, ddof=1)
                self.Y_train_mean = np.mean(self.Y_train, axis=0)
                self.Y_train_std = np.std(self.Y_train, ddof=1, axis=0)
                self.X_train = (self.X_train - self.X_train_mean) / self.X_train_std
                self.Y_train = (self.Y_train - self.Y_train_mean) / self.Y_train_std
                self.X_test = (self.X_test - self.X_train_mean) / self.X_train_std
                self.Y_test = (self.Y_test - self.Y_train_mean) / self.Y_train_std

    def fit(self, cov_type="nonrobust", is_output=True):
        '''
        Ordinary least square fit the linear regression model.
        params:
            cov_type: estimation method for covariance type
                "nonrobust": default, assume homoscedasticity
                "HC0": \Sigma_{ii} = \epsilon_i^2, where \epsilon_i is the residual of $i$-th observation. Good for large sample size.
                "HC1": \Sigma_{ii} = (n/(n-p))*\epsilon_i^2, where n is the number of observations and p is the number of features. Correction by degrees of freedom. Default correction.
                "HC2": \Sigma_{ii} = \epsilon_i^2/(1-h_{ii}), where h_{ii} is the leverage of $i$-th observation. Adjust for leverage, gives more weight to high-leverage points. Good for large sample size.
                "HC3": \Sigma_{ii} = \epsilon_i^2/(1-h_{ii})^2, where h_{ii} is the leverage of $i$-th observation. Adjust for leverage, gives more weight to high-leverage points. Best for small sample size.      
            is_output: whether to print the summary of the model
        '''
        self.cov_type = cov_type
        self.ols = sm.OLS(self.Y, sm.add_constant(self.X, prepend=True)).fit(cov_type=self.cov_type)
        if is_output:
            print(self.ols.summary())
            plt.figure(figsize=(6, 6))
            plt.subplot(2,1,1)
            plt.errorbar(self.X_columns, self.ols.params[1:], yerr=self.ols.bse[1:], capsize=5)
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xticks(rotation=45)
            plt.ylabel("Coefficient")
            plt.subplot(2,1,2)
            plt.bar(self.X_columns, self.ols.pvalues[1:], color=["green" if p < 0.01 else "orange" if p < 0.05 else "red" for p in self.ols.pvalues[1:]])
            plt.axhline(y=0.01, color='green', linestyle='--', label="p=0.01")
            plt.axhline(y=0.05, color='orange', linestyle='--', label="p=0.05")
            plt.legend(); plt.yscale("log"); plt.xticks(rotation=45)
            plt.ylabel("p-value"); plt.xlabel("Feature"); plt.tight_layout()

    def predict(self, X):
        '''
        Predict the target values for the given input features.
        params:
            X: feature matrix, shape (n, p)
        return:
            Y_pred: predicted target values, shape (n, 1)
        '''
        if not hasattr(self, "ols"):
            raise Exception("Fit the model first.")
        if self.is_normalize:
            X = (X - self.X_mean) / self.X_std
        Y_pred = self.ols.predict(sm.add_constant(X, prepend=True)).reshape(-1, 1)
        if self.is_normalize:
            Y_pred = Y_pred * self.Y_std + self.Y_mean
        return Y_pred

    def visualize_data(self, methods=["pandas_describe", "raw_data_plot", "boxplot", "PCA", "MDS", "tSNE", "UMAP"]):
        '''
        Visualize the data by:
            1. summary of the data
            2. plot each feature across observations (for time-series data)
            3. boxplot each feature across observations
            4. low-dimensional projection
        '''
        if "pandas_describe" in methods:
            # describe the data
            if self.is_normalize:
                df = pd.DataFrame(self.X * self.X_std[np.newaxis, :] + self.X_mean[np.newaxis, :], columns=self.X_columns)
                df["target"] = self.Y * self.Y_std[np.newaxis, :] + self.Y_mean[np.newaxis, :]
            else:
                df = pd.DataFrame(self.X, columns=self.X_columns)
                df["target"] = self.Y
            print("summary of the unnormalized data:")
            print(df.describe())

        if "raw_data_plot" in methods:
            # plot each feature across observations
            ncol = 3; nrow = (self.p + 1) // ncol + 1
            plt.figure(figsize=(4*ncol, 3*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                if self.is_normalize:
                    plt.plot(self.X[:, i] * self.X_std[i] + self.X_mean[i])
                else:
                    plt.plot(self.X[:, i])
                plt.ylabel(self.X_columns[i])
            plt.subplot(nrow, ncol, self.p + 1)
            if self.is_normalize:
                plt.plot(self.Y * self.Y_std + self.Y_mean, color="red")
            else:
                plt.plot(self.Y, color="red")
            plt.ylabel("target")
            plt.suptitle("Feature across (unnormalized) observations")
            plt.tight_layout()

        if "boxplot" in methods:
            # boxplot each feature across observations
            nrow = 2; ncol = self.p//nrow + 1
            plt.figure(figsize=(1.5*ncol, 2*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                if self.is_normalize:
                    plt.boxplot(self.X[:, i] * self.X_std[i] + self.X_mean[i], widths=0.5, showfliers=True)
                else:
                    plt.boxplot(self.X[:, i], widths=0.5, showfliers=True)
                plt.ylabel(self.X_columns[i])
            plt.subplot(nrow, ncol, self.p + 1)
            if self.is_normalize:
                plt.boxplot(self.Y * self.Y_std + self.Y_mean, widths=0.5, showfliers=True)
            else:
                plt.boxplot(self.Y, widths=0.5, showfliers=True)
            plt.ylabel("target")
            plt.suptitle("Boxplot of (unnormalized) features")
            plt.tight_layout()

        if self.p > 2 and any(["PCA" in methods, "MDS" in methods, "tSNE" in methods, "UMAP" in methods]):
            color_feature = self.Y.flatten()
            plt.figure(figsize=(12, 12))
            if "PCA" in methods:
                # projection by principal component analysis (PCA)
                U, S, VT = np.linalg.svd(self.X, full_matrices=False)
                V = VT.T; V = V[:, 0:2]
                X_trans = self.X.dot(V)
                plt.subplot(2,2,1)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=0.3)
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.title("Principal component analysis (PCA)")
                plt.colorbar(label=r"$Y$")

            if "MDS" in methods:
                # projection by multidimensional scaling (MDS)
                mds = sklearn.manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=None)
                X_trans = mds.fit_transform(self.X)
                plt.subplot(2,2,2)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=0.3)
                plt.xlabel("MDS Component 1")
                plt.ylabel("MDS Component 2")
                plt.title("Multidimensional Scaling (MDS)")
                plt.colorbar(label=r"$Y$")

            if "tSNE" in methods:
                # projection by t-distributed stochastic neighbor embedding (t-SNE)
                tsne = sklearn.manifold.TSNE(n_components=2, random_state=None)
                X_trans = tsne.fit_transform(self.X)
                plt.subplot(2,2,3)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=0.3)
                plt.xlabel("t-SNE Component 1")
                plt.ylabel("t-SNE Component 2")
                plt.title("t-distributed Stochastic Neighbor Embedding (t-SNE)")
                plt.colorbar(label=r"$Y$")
            
            if "UMAP" in methods:
                # projection by Uniform Manifold Approximation and Projection (UMAP)
                umap_ = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean")
                X_trans = umap_.fit_transform(self.X)
                plt.subplot(2,2,4)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=0.3)
                plt.xlabel("UMAP Component 1")
                plt.ylabel("UMAP Component 2")
                plt.title("Uniform Manifold Approximation and Projection (UMAP)")
                plt.colorbar(label=r"$Y$")
            plt.suptitle("Visualization of data by low-dimensional projection")
            plt.tight_layout()

    def nonlinearity(self, smoother_type="polynomial", method=["RESET", "residual", "partial_residual", "add_variable", "interaction_term"]):
        '''
        Diagonostic analysis on non-linearity.
        Analysis include:
            (1) Ramsey reset test
                y = X\beta + \gamma_1 \hat{y}^2 + \gamma_2 \hat{y}^3 + ... + \varepsilon. 
                H_0: \gamma_1 = \gamma_2 = ... = \gamma_n = 0

            (2) Partial residual plot 
                r_{ij}^{partial} = \hat{\varepsilon} + \hat{\beta_j}X_{ij}. 
                Show r_{ij}^{partial} vs X_{ij} for j=1,2,..., p. 
                If the plot shows curvature, it suggests non-linearity. 

            (3) Add variable plot
                r_y vs r_{x_j}, where r_y is the residual for regressing y on all predictors except x_j, and r_{x_j} is the residual for regressing x_j on all other predictors. 
                - A linear pattern indicates a strong partial linear relationship between x_j and y;
                - A flat or noisy pattern suggests x_j adds little value to the model;
                - Curvature may indicate nonlinearity or model misspecification.

            (4) Compare with models that include interaction term
                Add term x_i*x_j in model and compare the model metric
                    p-value of F-statistic
                        H_0: small model. 
                        If p is small, we reject the small model and proceed with the larger model. 
                    AIC, BIC, R^2 (out of sample): accept large model when we observe a significant reduction
        
        params:
            smoother_type: type for smoother in the plots. See self._smoother for details.

        returns:
            self.nonlinearity_test: dict
                - "RESET" -- RESET test p-value
                - "interaction_term_metric" -- [p-value of F-statistics, AIC, BIC, out-of-sample R^2]
        '''
        if not hasattr(self, "ols"):
            raise Exception("Fit the model first.")
        self.nonlinearity_test = {}

        # Ramsey reset test (RESET)
        if "RESET" in method:
            result = statsmodels.stats.diagnostic.linear_reset(self.ols, power=3, use_f=False, cov_type=self.cov_type)
            self.nonlinearity_test['RESET'] = result.pvalue
            print("RESET test statistic: %.3f, p-value: %.4f" % (result.statistic, result.pvalue))

        # plot the residuals vs features
        if "residual" in method:
            ncol = 4; nrow = (self.p + 1) // ncol + 1
            plt.figure(figsize=(3*ncol, 3*nrow))
            plt.subplot(nrow, ncol, 1)
            Y_pred = self.ols.predict(sm.add_constant(self.X, prepend=True))
            residual = self.ols.resid
            plt.scatter(Y_pred, np.power(residual, 2), s=1)
            smoother = self._smoother(Y_pred, np.power(residual, 2), type=smoother_type)
            plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
            plt.xlabel(r"$\hat{Y}$"); plt.ylabel(r"$\hat{\varepsilon}^2$"); plt.legend()

            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 2)
                plt.scatter(self.X[:, i], np.power(residual, 2), s=1)
                smoother = self._smoother(self.X[:, i], np.power(residual, 2), type=smoother_type)
                plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
                plt.xlabel(self.X_columns[i]); plt.ylabel(r"$\hat{\varepsilon}^2$"); plt.legend()
            plt.suptitle("Residuals vs features")
            plt.tight_layout()

        # partial residual plot
        if "partial_residual" in method:
            plt.figure(figsize=(3*ncol, 3*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 2)
                partial_residual = self.ols.resid + self.ols.params[i+1] * self.X[:, i]
                plt.scatter(self.X[:, i], partial_residual, s=1)
                smoother = self._smoother(self.X[:, i], partial_residual, type=smoother_type)
                plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
                plt.xlabel(self.X_columns[i]); plt.ylabel(r"$\hat{\varepsilon}^2$"); plt.legend()
            plt.suptitle("Partial residuals vs features")
            plt.tight_layout()

        # add variable plots
        if "add_variable" in method:
            plt.figure(figsize=(3*ncol, 3*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 2)
                X_temp = self.X[:, np.arange(self.p)!=i].reshape((self.n, -1))
                linreg = sklearn.linear_model.LinearRegression()
                linreg.fit(X_temp, self.Y)
                r_y = self.Y - linreg.predict(X_temp).reshape(-1, 1)
                linreg = sklearn.linear_model.LinearRegression()
                linreg.fit(X_temp, self.X[:, i].reshape(-1, 1))
                r_x = self.X[:, i].reshape(-1, 1) - linreg.predict(X_temp).reshape(-1, 1)
                plt.scatter(r_x, r_y, s=1)
                smoother = self._smoother(r_x, r_y, type=smoother_type)
                plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
                plt.xlabel(r"$r_x: x_j \sim x_{(-j)}$"); plt.ylabel(r"$r_y: y \sim X_{(-j)}$"); plt.legend()
                plt.title(self.X_columns[i])
            plt.suptitle("Add variable plot")
            plt.tight_layout()

        # compare with polynomial model
        if "interaction_term" in method:
            F_hist = np.zeros((self.p, self.p)); F_hist[:] = np.nan
            AIC_hist = np.zeros((self.p, self.p)); AIC_hist[:] = np.nan
            BIC_hist = np.zeros((self.p, self.p)); BIC_hist[:] = np.nan
            CV_error = np.zeros((self.p, self.p)); CV_error[:] = np.nan
            RSS_benchmark = self.ols.ssr
            ols = sm.OLS(self.Y_train, sm.add_constant(self.X_train, prepend=True)).fit(cov_type=self.cov_type)
            Y_pred = ols.predict(sm.add_constant(self.X_test, prepend=True)).reshape(-1, 1)
            CV_error_benchmark = np.sqrt(np.mean((self.Y_test - Y_pred)**2))

            for i in range(self.p):
                for j in range(self.p):
                    X_temp = np.concatenate([self.X, (self.X[:, i]*self.X[:, j]).reshape((-1, 1))], axis=1)
                    ols = sm.OLS(self.Y, sm.add_constant(X_temp, prepend=True)).fit(cov_type=self.cov_type)
                    F_stats = (self.n-self.p-1)*(RSS_benchmark - ols.ssr)/ols.ssr
                    F_hist[i, j] = 1-scipy.stats.f.cdf(F_stats, 1, self.n-self.p-1)
                    AIC_hist[i, j] = ols.aic-self.ols.aic
                    BIC_hist[i, j] = ols.bic-self.ols.bic
                    X_temp = np.concatenate([self.X_train, (self.X_train[:, i]*self.X_train[:, j]).reshape((-1, 1))], axis=1)
                    ols = sm.OLS(self.Y_train, sm.add_constant(X_temp, prepend=True)).fit(cov_type=self.cov_type)
                    X_test_temp = np.concatenate([self.X_test, (self.X_test[:, i]*self.X_test[:, j]).reshape((-1, 1))], axis=1)
                    Y_pred = ols.predict(sm.add_constant(X_test_temp, prepend=True)).reshape(-1, 1)
                    CV_error[i, j] = (np.sqrt(np.mean((self.Y_test - Y_pred)**2)) - CV_error_benchmark)/CV_error_benchmark

            self.nonlinearity_test["interaction_term_metric"] = [F_hist, AIC_hist, BIC_hist, CV_error]

            plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            colors = ['red', 'yellow', 'green']  # for <0.01, 0.01–0.05, >0.05
            bounds = [0, 0.01, 0.05, 1]
            cmap = matplotlib.colors.ListedColormap(colors)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            plt.imshow(F_hist, cmap=cmap, norm=norm)
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.yticks(range(self.p), self.X_columns)
            plt.colorbar()
            plt.title(r"p-value of $F$-statistic")

            plt.subplot(2, 2, 2)
            plt.imshow(AIC_hist, cmap='Reds_r', vmax=0, aspect='auto')
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.yticks(range(self.p), self.X_columns)
            plt.colorbar()
            plt.title(r"$AIC - AIC_{bench}$")

            plt.subplot(2, 2, 3)    
            plt.imshow(BIC_hist, cmap='Reds_r', vmax=0, aspect='auto')
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.yticks(range(self.p), self.X_columns)
            plt.colorbar()
            plt.title(r"$BIC - BIC_{bench}$")

            plt.subplot(2, 2, 4)
            plt.imshow(CV_error, cmap='Reds_r', vmax=0, aspect='auto')
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.yticks(range(self.p), self.X_columns)
            plt.colorbar()
            plt.title(r"$(CV_{error} - CV_{error, bench})/CV_{error, bench}$")
            plt.suptitle("Polynomial model comparison")
            plt.tight_layout()  


        return self.nonlinearity_test

    def outlier(self, threshold="strict", is_output=True):
        '''
        Diagnostic analysis on outliers.
        Analysis include:
            (1) Leverage: H_{ii}, where H = X(X^TX)^{-1}X^TY
            (2) Internal studentized residual 
                r_i = \frac{\hat{\varepsilon}_i}{\hat{\sigma} \sqrt{1 - h_i}} such that Var(r_i) = 1   
                \hat{\Sigma} include observation i             
            (3) External studentized residual 
                r_i = \frac{\hat{\varepsilon}_i}{\hat{\sigma}_{(i)} \sqrt{1 - h_i}} such that Var(r_i) = 1  
                \hat{\sigma}_{(i)} exclude observation i
            (4) Cook distance measures how much the entire fitted regression model would change if a single observation were removed.
                D_i = \frac{(\hat{\beta} - \hat{\beta}_{(i)})^\top X^\top X (\hat{\beta} - \hat{\beta}_{(i)})}{p \hat{\sigma}^2} 
                    = \frac{\sum_{j=1}^N (\hat{y}_j - \hat{y}_{j(i)})^2}{p \hat{\sigma}^2}

        params:
            threshold: str, defines the threshold for outlier detection.
                - "strict": 2 standard deviations
                - "loose": 3 standard deviations
            is_output: whether to print the summary of the outliers

        return:
            self.outlier_test: dict
                - "external_studentized_residuals": observation index for outliers by external studentized residuals
                - "p-value": observation index for outliers by p-value of external studentized residuals
                - "leverage": observation index for outliers for outliers by leverage
                - "cooks_distance": observation index for outliers by Cook's distance
                - "summary": summary of outliers, shape (n, 4), where each column indicates whether the observation is an outlier by the corresponding method
                - "outlier_idx": observation index for outliers by any of the methods
        '''

        if not hasattr(self, "ols"):
            raise Exception("Fit the model first.")

        self.outlier_test = {}
        leverage = self.ols.get_influence().hat_matrix_diag
        studentized_residuals_internal = self.ols.get_influence().resid_studentized_internal
        studentized_residuals_external = self.ols.get_influence().resid_studentized_external
        cook_distance = self.ols.get_influence().cooks_distance[0]
        p_value = [1-scipy.stats.t.cdf(np.abs(studentized_residuals_external[i]), self.n-self.p-1) for i in range(self.n)]

        self.outlier_test["external_studentized_residuals"] = np.where(np.abs(studentized_residuals_external) > (2 if threshold=="strict" else 3))[0]
        self.outlier_test["p_value"] = np.where(np.array(p_value) < 0.05)[0]
        self.outlier_test["leverage"] = np.where(leverage > (2*self.p/self.n) if threshold=="strict" else (3*self.p/self.n))[0]
        self.outlier_test["cooks_distance"] = np.where(cook_distance > (4/(self.n-self.p-1) if threshold=="strict" else 1))[0]

        outlier_summary = np.zeros((self.n, 4))
        outlier_summary[self.outlier_test["external_studentized_residuals"], 0] = 1
        outlier_summary[self.outlier_test["p_value"], 1] = 1
        outlier_summary[self.outlier_test["leverage"], 2] = 1
        outlier_summary[self.outlier_test["cooks_distance"], 3] = 1
        self.outlier_test["summary"] = outlier_summary
        self.outlier_test["outlier_idx"] = np.where(np.sum(outlier_summary, axis=1) > 0)[0]

        if is_output:
            plt.figure(figsize=(6, 8))
            plt.subplot(2, 1, 1)
            plt.scatter(studentized_residuals_external, leverage, s=1)
            x_min, x_max = plt.gca().get_xlim()
            y_min, y_max = plt.gca().get_ylim()
            plt.hlines(y=2*self.p/self.n, xmin=x_min, xmax=x_max, color='orange', linestyle='--')
            plt.hlines(y=3*self.p/self.n, xmin=x_min, xmax=x_max, color='red', linestyle='--')
            plt.vlines(x=[-2, 2], ymin=y_min, ymax=y_max, color='orange', linestyle='--')
            plt.vlines(x=[-3, 3], ymin=y_min, ymax=y_max, color='red', linestyle='--')
            plt.fill_between([-2, 2], 0, 2*self.p/self.n, color="green", alpha=0.2, label="not outlier")
            plt.legend()
            plt.xlabel("external studentized residuals"); plt.ylabel("leverage")

            plt.subplot(4, 1, 3)
            plt.imshow(outlier_summary.T, cmap='Reds', vmin=0, vmax=1, aspect='auto')
            plt.colorbar()
            plt.yticks(range(4), ["stu. res.", "p-value", "leverage", "cooks dist."])
            plt.title("Outlier summary")

            plt.subplot(4, 1, 4)
            plt.scatter(range(self.n), np.sum(outlier_summary, axis=1), s=5, color="red")
            plt.xlim(0, self.n-1)
            plt.ylabel("Number of \n indicated outliers")

            plt.suptitle("Outlier Detection Plot")
            plt.tight_layout()
        
        return self.outlier_test

    def colinearity(self, method=["pairwise_scatter", "pairwise_R2_corr", "VIF", "eigenvalue"]):
        '''
        Diagnostic analysis on colinearity.
        Analysis include:
            (1) Pair-wise scatter plot and R^2 of features
            (2) Pair-wise correlation matrix

            (3) Variance inflation factor (VIF)
                VIF = 1/(1-R^2), where R^2 is the R^2 of regressing x_i on all other features.
                - VIF < 1: acceptable
                - 1 <= VIF < 5: caution
                - VIF >= 5: warning

            (4) Eigenvalue spectra of correlation matrix. 
                Conditional number = max(eigenvalue)/min(eigenvalue)
                - CN < 10: acceptable
                - 10 <= CN < 30: caution
                - CN >= 30: warning

        return:
            self.colinearity_test: dict
                - "pairwise_R2": pair-wise R^2 of features, shape (p, p)
                - "correlation": pair-wise correlation matrix, shape (p, p)
                - "variance_inflation_factor": variance inflation factor of features, shape (p,)
                - "eigenvalue": eigenvalue spectra of correlation matrix, shape (p,)
        '''
        self.colinearity_test = {}
        if "pairwise_scatter" in method:
            # pair-wise scatter plot of features
            R2 = np.zeros((self.p, self.p)); R2[:] = np.nan
            plt.figure(figsize=(3*self.p, 3*self.p))
            for i in range(self.p):
                for j in range(self.p):
                    if i != j:
                        plt.subplot(self.p, self.p, i * self.p + j + 1)
                        plt.scatter(self.X[:, i], self.X[:, j], s=1)
                        linreg = sklearn.linear_model.LinearRegression()
                        linreg.fit(self.X[:, i].reshape(-1, 1), self.X[:, j])
                        R2[i, j] = linreg.score(self.X[:, i].reshape(-1, 1), self.X[:, j])
                        plt.plot(self.X[:, i], linreg.predict(self.X[:, i].reshape(-1, 1)), color="red", linestyle="--", label = f"y={linreg.intercept_:.3f}+{linreg.coef_[0]:.3f}x\nR^2={R2[i, j]:.2f}")
                        plt.xlabel(self.X_columns[i])
                        plt.ylabel(self.X_columns[j])
                        plt.legend()
            plt.suptitle("Pair-wise feature scatter plot")
            plt.tight_layout()
            self.colinearity_test["pairwise_R2"] = R2

        # plot pairwise R2 and correlation matrix
        R2 = np.zeros((self.p, self.p)); R2[:] = np.nan
        for i in range(self.p):
            for j in range(self.p):
                if i != j:
                    linreg = sklearn.linear_model.LinearRegression()
                    linreg.fit(self.X[:, i].reshape(-1, 1), self.X[:, j])
                    R2[i, j] = linreg.score(self.X[:, i].reshape(-1, 1), self.X[:, j])
        corr = np.corrcoef(self.X, rowvar=False)
        self.colinearity_test["pairwise_R2"] = R2
        self.colinearity_test["correlation"] = corr

        if "pairwise_R2_corr" in method:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title(r"Pair-wise $R^2$")
            plt.imshow(R2, cmap='Reds', vmin=0, vmax=1, aspect='auto')
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.yticks(range(self.p), self.X_columns)
            plt.title("Feature pair-wise R2 matrix")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Correlation matrix")
            plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.yticks(range(self.p), self.X_columns)
            plt.title("Feature correlation matrix")
            plt.colorbar()
            plt.tight_layout()

        # calculate variance inflation factors
        variance_inflation_factor = np.zeros(self.p)
        for i in range(self.p):
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(self.X[:, np.arange(self.p) != i], self.X[:, i])
            R2 = linreg.score(self.X[:, np.arange(self.p) != i], self.X[:, i])
            variance_inflation_factor[i] = 1 / (1 - R2)
        self.colinearity_test["variance_inflation_factor"] = variance_inflation_factor

        if "VIF" in method:
            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.bar(range(self.p), variance_inflation_factor, color=["green" if vif < 5 else "orange" if vif < 10 else "red" for vif in variance_inflation_factor])
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.ylabel("variance inflation factor")
            plt.axhline(y=10, color='r', linestyle='--', label="Warning")
            plt.axhline(y=5, color='orange', linestyle='--', label="Caution")
            plt.axhline(y=2.5, color='green', linestyle='--', label="Acceptable")
            plt.yscale("log")
            plt.title("Variance Inflation Factor")
            plt.legend()

        # correlation matrix structure
        if "eigenvalue" in method:
            plt.subplot(1, 2, 2)
            S = np.linalg.svd(self.X, compute_uv=False)
            self.colinearity_test["eigenvalue"] = np.abs(S)**2/(self.n-1)
            plt.plot(range(min(self.n, self.p)), np.abs(S)**2/(self.n-1), "o-")
            plt.fill_between(range(min(self.n, self.p)), (1-np.sqrt(self.p/self.n))**2, (1+np.sqrt(self.p/self.n))**2, color="red", alpha=0.2, label="Marchenko-Pastur")
            plt.xlabel("index")
            plt.ylabel("eigenvalue of correlation matrix")
            plt.legend(title="condition number: %.2f" % (np.max(np.abs(S)**2/(self.n-1))/np.min(np.abs(S)**2/(self.n-1))))
            plt.title("correlation matrix \n eigenvalue spectra")
            plt.tight_layout()

        return self.colinearity_test

    def homoscedasticity(self, method=["residual_vs_feature_plot", "test", "Box-Cox", "Yeo-Johnson"]):     
        '''
        Diagnostic analysis on homoscedasticity.
        Analysis include:
            (1) Plot the residuals vs features
                - If the plot shows a pattern, it suggests non-constant variance.
                - If the plot shows a random scatter, it suggests constant variance.

            (2) Breusch-Pagan test
                \hat{\varepsilon}^2 = X\gamma + v
                H_0: \gamma = 0, i.e., homoscedasticity

            (3) Park test
                \log(\hat{\varepsilon}^2) = \log(X)\gamma + v
                H_0: \gamma = 0, i.e., homoscedasticity

            (4) White test
                \hat{\varepsilon}^2 = X\gamma + X_iX_j\beta + v
                H_0: \gamma = 0, i.e., homoscedasticity

            (5) Goldfeld-Quandt test (non-parametric)
                H_0: \sigma^2_{large}/\sigma^2_{small} = 1, i.e., homoscedasticity
                - Sort the data by the feature suspected to be heteroscedastic
                - Split the data into two groups: large and small
                - Calculate the ratio of the variances of the two groups

            (6) Harrison-McCabe test
                H_0: \sigma^2_{large}/\sigma^2_{small} = 1, i.e., homoscedasticity
                - Sort the data by the residuals
                - Split the data into two groups: large and small
                - Calculate the ratio of the variances of the two groups

            (7) Box-Cox transformation
                Find the optimal Box-Cox transformation parameter lambda that maximizes the log-likelihood function for normal residuals.
                - If lambda = 1, it suggests no transformation.

            (8) Yeo-Johnson transformation
                Find the optimal Yeo-Johnson transformation parameter lambda that maximizes the log-likelihood function for normal residuals.
                - If lambda = 1, it suggests no transformation.

        return:
            self.homoscedasticity_test: dict
                - "Breusch-Pagan": p-value of Breusch-Pagan test
                - "Park": p-value of Park test
                - "White": p-value of White test
                - "Goldfeld-Quandt": p-value of Goldfeld-Quandt test
                - "Harrison-McCabe": p-value of Harrison-McCabe test
                - "Box-Cox": optimal Box-Cox transformation parameter lambda
                - "Yeo-Johnson": optimal Yeo-Johnson transformation parameter lambda
        '''
        if not hasattr(self, "ols"):
            raise Exception("Fit the model first.")
        residuals = self.ols.resid.reshape(-1, 1)
        self.homoscedasticity_test = {}

        if "residual_vs_feature_plot" in method:
            # plot the residuals vs features
            ncol = 4
            nrow = (self.p + 1) // ncol + 1
            plt.figure(figsize=(3*ncol, 3*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                plt.scatter(self.X[:, i], np.power(residuals, 2), s=1)
                linreg = sklearn.linear_model.LinearRegression()
                linreg.fit(self.X[:, i].reshape(-1, 1), np.power(residuals, 2))
                R2 = linreg.score(self.X[:, i].reshape(-1, 1), np.power(residuals, 2))
                plt.plot(self.X[:, i], linreg.predict(self.X[:, i].reshape(-1, 1)), color="red", linestyle="--", label = r"$y=%.2f+%.2fx$"% (linreg.intercept_, linreg.coef_[0]) + "\n $R^2$=%.2f" % R2)
                plt.legend()
                plt.xlabel(self.X_columns[i])
                plt.ylabel(r"$\hat{\varepsilon}^2$")
                plt.axhline(y=0, color='gray', linestyle='--')
            plt.suptitle("Homoscedasticity: residuals vs X and Y")
            plt.tight_layout()

        if "test" in method:
            test_name = []; test_p_value = []
            # Heteroscedasticity test -- Breuch-Pagan test
            bp_test_statistic, bp_p_value, _, _ = sms.het_breuschpagan(residuals, sm.add_constant(self.X, prepend=True), robust=True)
            test_name.append("Breusch-Pagan\n" + r"$\varepsilon^2 \sim X$")
            test_p_value.append(bp_p_value)
            self.homoscedasticity_test["Breusch-Pagan"] = bp_p_value
            #print("Breusch-Pagan test statistic: %.4f, p-value: %.4f" % (bp_test_statistic, bp_p_value))

            # Heteroscedasticity test -- Park test
            if self.X.flatten().min() <= 0:
                park_test_statistic, park_test_pvalue = np.nan, np.nan
            else:
                model = sm.OLS(np.log(residuals**2), np.log(sm.add_constant(self.X, prepend=True))).fit()
                park_test_statistic, park_test_pvalue = model.fvalue, model.f_pvalue
            test_name.append("Park\n" + r"$\log(\varepsilon^2) \sim \log(X)$")
            test_p_value.append(park_test_pvalue)
            self.homoscedasticity_test["Park"] = park_test_pvalue
            #print("Park test statistic: %.4f, p-value: %.4f" % (park_test_statistic, park_test_pvalue))

            # Heteroscedasticity test -- White test
            white_test_statistic, white_p_value, _, _ = sms.het_white(residuals, sm.add_constant(self.X, prepend=True))
            test_name.append("White\n" + r"$\varepsilon^2 \sim X + X_iX_j$")
            test_p_value.append(white_p_value)
            self.homoscedasticity_test["White"] = white_p_value
            #print("White test statistic: %.4f, p-value: %.4f" % (white_test_statistic, white_p_value))

            # Heteroscedasticity test -- Goldfeld-Quantdt test
            gf_p_value = 1; factor_idx = None
            for i in range(self.p):
                gf_test_statistic_temp, gf_p_value_temp, _ = sms.het_goldfeldquandt(residuals.flatten(), self.X[:, i].reshape((-1, 1)))
                if gf_p_value_temp < gf_p_value:
                    gf_test_statistic, gf_p_value = gf_test_statistic_temp, gf_p_value_temp
                    factor_idx = i

            #gf_test_statistic, gf_p_value, _ = sms.het_goldfeldquandt(residuals, sm.add_constant(self.X, prepend=True))
            test_name.append("Goldfeld-Quantdt \n" + r"$\sigma^2_{large}/\sigma^2_{small}$")
            test_p_value.append(gf_p_value)
            self.homoscedasticity_test["Goldfeld-Quantdt"] = gf_p_value
            #print("Goldfeld test (sorted by " + self.X_columns[factor_idx] + ") statistic: %.4f, p-value: %.4f" % (gf_test_statistic, gf_p_value) )

            # Heteroscedasticity test -- Harrison-McCabe test
            sorted_indices = np.argsort(residuals.flatten())
            sorted_residuals = residuals[sorted_indices, :]
            mid = len(sorted_residuals) // 2
            group1 = sorted_residuals[:mid, :]; var1 = np.var(group1, ddof=1)
            group2 = sorted_residuals[mid:, :]; var2 = np.var(group2, ddof=1)
            F = max(var1, var2) / min(var1, var2)
            df1 = len(group1) - self.p; df2 = len(group2) - self.p
            p_value = 1 - scipy.stats.f.cdf(F, df1, df2)
            hm_test_statistic, hm_test_pvalue = F, p_value
            test_name.append("Harrison-McCabe \n" + r"$\sigma^2_{large}/\sigma^2_{small}$")
            test_p_value.append(hm_test_pvalue)
            self.homoscedasticity_test["Harrison-McCabe"] = hm_test_pvalue
            #print("Harrison-McCabe test statistic: %.4f, p-value: %.4f" % (hm_test_statistic, hm_test_pvalue))

            plt.figure(figsize=(8,3))
            plt.bar(test_name, test_p_value, color=["green" if p>0.05 else "orange" if p>0.01 else "red" for p in test_p_value])
            x_min = plt.gca().get_xlim()[0]; x_max = plt.gca().get_xlim()[1]
            plt.hlines(y=0.05, xmin=x_min, xmax=x_max, color='green', linestyle='--', label="homoscedasticity")
            plt.hlines(y=0.01, xmin=x_min, xmax=x_max, color='orange', linestyle='--', label="caution")
            plt.hlines(y=0.001, xmin=x_min, xmax=x_max, color='red', linestyle='--', label="heteroscedasticity")
            plt.legend()
            plt.ylabel("p-value")
            plt.yscale("log")
            plt.ylim(max(plt.gca().get_ylim()[0], -5), 1)
            plt.title("Homoscedasticity test")
            plt.tight_layout()

        if "Box-Cox" in method:
            # Box-Cox transformation
            if any(self.Y <= 0):
                print("Box-Cox transformation is not applicable because Y exists negative values.")
            else:
                lambda_ = np.linspace(-10, 10, 1000)
                log_likelihood = np.zeros(lambda_.shape)
                for i in range(len(lambda_)):
                    Y_trans = scipy.stats.boxcox(self.Y, lmbda=lambda_[i])
                    ols = sm.OLS(Y_trans, sm.add_constant(self.X, prepend=True)).fit()
                    residual = Y_trans - ols.predict(sm.add_constant(self.X, prepend=True)).reshape(-1, 1)
                    log_likelihood[i] = -0.5*self.n*np.log(np.mean(np.power(residual, 2))) + (lambda_[i] - 1)*np.sum(np.log(self.Y))
                plt.figure(figsize=(6, 3))
                plt.plot(lambda_, log_likelihood)
                plt.vlines(lambda_[np.argmax(log_likelihood)], ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label=r"$\lambda^*$"+"=%.3f" % lambda_[np.argmax(log_likelihood)])
                plt.xlabel("power"); plt.ylabel("log-likelihood")
                plt.legend()
                plt.title("Box-Cox transformation")
                self.homoscedasticity_test["Box-Cox"] = lambda_[np.argmax(log_likelihood)]

        if "Yeo-Johnson" in method:
            # Yeo-Johnson transformation
            lambda_ = np.linspace(-10, 10, 1000)
            log_likelihood = np.zeros(lambda_.shape)
            for i in range(len(lambda_)):
                Y_trans = scipy.stats.yeojohnson(self.Y, lmbda=lambda_[i])
                ols = sm.OLS(Y_trans, sm.add_constant(self.X, prepend=True)).fit()
                residual = Y_trans - ols.predict(sm.add_constant(self.X, prepend=True)).reshape(-1, 1)
                pos_idx = np.where(self.Y >= 0)[0]; neg_idx = np.where(self.Y < 0)[0]
                log_likelihood[i] = -0.5*self.n*np.log(np.mean(np.power(residual, 2))) + (lambda_[i]-1)*np.sum(np.log(self.Y[pos_idx, :]+1)) + (1-lambda_[i])*np.sum(np.log(-self.Y[neg_idx, :]+1))
            plt.figure(figsize=(6, 3))
            plt.plot(lambda_, log_likelihood)
            plt.vlines(lambda_[np.argmax(log_likelihood)], ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label=r"$\lambda^*$"+"=%.3f" % lambda_[np.argmax(log_likelihood)])
            plt.xlabel("power"); plt.ylabel("log-likelihood")
            plt.legend()
            plt.title("Yeo-Johnson transformation")
            self.homoscedasticity_test["Yeo-Johnson"] = lambda_[np.argmax(log_likelihood)]

    def test_residual_normality_independence(self, method=["Q-Q plot", "test"], lag_max=1):
        '''
        Diagnostic analysis on residual normality and independence.
        Analysis include:
        - Q-Q plot: plot the quantiles of the residuals against the quantiles of a normal distribution.
        - Shapiro-Wilk test (normality test)
        - Anderson-Darling test (normality test)
        - Jarque-Bera test (normality test)
        - Durbin-Watson test (independence test)
            \varepsilon_i = \rho \varepsilon_{i-1} + v_i, where v_i \sim N(0, \sigma^2)
            H_0: \rho = 0, i.e., independence
        - Breusch-Godfrey test (independence test)
            \varepsilon_i = \rho_1 \varepsilon_{i-1} + \rho_2 \varepsilon_{i-2} + ... + \rho_k \varepsilon_{i-k} + v_i, where v_i \sim N(0, \sigma^2)
            H_0: \rho_1 = \rho_2 = ... = \rho_k = 0, i.e., independence        
        - Ljung-Box test (independence test)
            H_0: \rho_1 = \rho_2 = ... = \rho_k = 0, i.e., independence
        - Wald-Wolfowitz test (independence test, run test)
            H_0: \rho_1 = \rho_2 = ... = \rho_k = 0, i.e., independence
            - The test statistic is the number of runs in the sequence of residuals.
            - A run is a sequence of consecutive identical signs (positive or negative) in the residuals. 

        return:
            self.residual_normal_independent_test: dict
                - "Shapiro-Wilk": p-value of Shapiro-Wilk test
                - "Anderson-Darling": p-value of Anderson-Darling test
                - "Jarque-Bera": p-value of Jarque-Bera test
                - "Durbin-Watson": p-value of Durbin-Watson test
                - "Breusch-Godfrey": p-value of Breusch-Godfrey test
                - "Ljung-Box": p-value of Ljung-Box test
                - "Wald-Wolfowitz": p-value of Wald-Wolfowitz test          
        '''
        if not hasattr(self, "ols"):
            raise Exception("Fit the model first.")
        self.residual_normal_independent_test = {}
        residuals = self.ols.resid.reshape(-1, 1)

        if "Q-Q plot" in method:
            # Q-Q plot
            plt.figure(figsize=(4, 4))
            scipy.stats.probplot(residuals.flatten(), dist="norm", plot=plt)
            plt.title("Q-Q plot of residuals")
            plt.xlabel("theoretical quantiles"); plt.ylabel("sample quantiles")

        if "test" in method:
            # Shapiro-Wilk test for normality
            shapiro_test_statistic, shapiro_p_value = scipy.stats.shapiro(residuals.flatten())
            self.residual_normal_independent_test["Shapiro-Wilk"] = shapiro_p_value

            # Anderson-Darling test for normality
            result = scipy.stats.anderson(residuals.flatten(), dist="norm")
            anderson_p_value = np.interp(result.statistic, result.critical_values, result.significance_level)/100
            self.residual_normal_independent_test["Anderson-Darling"] = anderson_p_value

            # Jarque-Bera test for normality
            jarque_bera_test_statistic, jarque_bera_p_value = scipy.stats.jarque_bera(residuals.flatten())
            self.residual_normal_independent_test["Jarque-Bera"] = jarque_bera_p_value

            # Durbin-Watson test for independence
            durbin_watson_test_statistic = float(sms.durbin_watson(residuals))
            if np.abs(durbin_watson_test_statistic-2) > 1:
                durbin_watson_p_value = 0.01
            elif np.abs(durbin_watson_test_statistic-2) > 0.2:
                durbin_watson_p_value = 0.05
            else:
                durbin_watson_p_value = 0.5
            self.residual_normal_independent_test["Durbin-Watson"] = durbin_watson_p_value

            # Breusch-Godfrey test for independence
            result = sms.acorr_breusch_godfrey(self.ols, nlags=lag_max, store=False)
            breusch_godfrey_test_statistic, breusch_godfrey_p_value = result[0], result[1]
            self.residual_normal_independent_test["Breusch-Godfrey"] = breusch_godfrey_p_value

            # Ljung-Box test for independence
            result = sms.acorr_ljungbox(residuals, lags=lag_max, return_df=False).to_numpy()
            self.residual_normal_independent_test["Ljung-Box"] = np.min(result[:, 1])
            
            # Wald-Wolfowitz test for independence
            def wald_wolfowitz_test(data):
                median = np.median(data)
                signs = ['+' if x > median else '-' for x in data]
                runs = 1
                for i in range(1, len(signs)):
                    if signs[i] != signs[i-1]:
                        runs += 1
                n_pos = signs.count('+'); n_neg = signs.count('-')

                expected_runs = ((2 * n_pos * n_neg) / (n_pos + n_neg)) + 1
                std_runs = np.sqrt((2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) /
                                (((n_pos + n_neg)**2) * (n_pos + n_neg - 1)))

                z = (runs - expected_runs) / std_runs
                p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
                return z, p_value
            wald_wolfowitz_test_statistic, wald_wolfowitz_p_value = wald_wolfowitz_test(residuals.flatten())
            self.residual_normal_independent_test["Wald-Wolfowitz"] = wald_wolfowitz_p_value

            plt.figure(figsize=(8, 3))
            key_list = list(self.residual_normal_independent_test.keys())
            plt.plot(key_list, [self.residual_normal_independent_test[key] for key in key_list], "-o")
            plt.fill_between(key_list[0:3], 0.05, 1, alpha=0.4, color="green", label="normality")
            plt.fill_between(key_list[0:3], 0, 0.05, alpha=0.4, color="red", label="non-normality")
            plt.fill_between(key_list[3:], 0.05, 1, alpha=0.2, color="green", label="independence")
            plt.fill_between(key_list[3:], 0, 0.05, alpha=0.2, color="red", label="non-independence")
            plt.legend(ncol=2); plt.ylabel("p-value"); plt.xticks(rotation=45)
            plt.title("Residual normality and independence test")
            plt.tight_layout()

        return self.residual_normal_independent_test

    def feature_target_correlation(self):
        '''
        Diagnostic analysis on feature-target correlation.
        Analysis include:
            (1) Pair-wise scatter plot and R^2 of features vs target
            (2) Pair-wise correlation matrix of features vs target
        '''
        ncol = 4
        nrow = (self.p + 1) // ncol + 1
        plt.figure(figsize=(3*ncol, 3*nrow))
        R2_list = []
        for i in range(self.p):
            plt.subplot(nrow, ncol, i + 1)
            plt.scatter(self.X[:, i], self.Y, s=1)
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(self.X[:, i].reshape(-1, 1), self.Y)
            R2 = linreg.score(self.X[:, i].reshape(-1, 1), self.Y)
            R2_list.append(R2)
            plt.plot(self.X[:, i], linreg.predict(self.X[:, i].reshape(-1, 1)), color="red", linestyle="--", label = f"y={linreg.intercept_[0]:.3f}+{linreg.coef_[0][0]:.3f}x \n R^2={R2:.2f}")
            plt.legend()
            plt.xlabel(self.X_columns[i])
            plt.ylabel(r"$Y$")

        plt.subplot(nrow, ncol, self.p + 1)
        plt.plot(range(self.p), R2_list, "o-", color="red")
        plt.xlabel("feature index")
        plt.ylabel(r"$R^2$")
        plt.suptitle("Feature vs Target")
        plt.tight_layout()

    def feature_selection_all(self, is_plot=True):
        '''
        Feature selection using different methods:
            (1) Best subset selection (self.feature_selection_best_subset)
            (2) Forward stepwise selection (self.feature_selection_forward_stepwise)
            (3) Ridge regression (self.feature_selection_ridge_lasso)
            (4) Lasso regression (self.feature_selection_ridge_lasso)
            (5) Least angle regression (self.feature_selection_least_angle_regression)

        return:
            returns of all feature selection methods above 
        '''          
        
        self.selected_feature = collections.defaultdict(list)
        self.feature_selection_best_subset(criterion="R2 (out-of-sample)", is_plot=is_plot)
        self.feature_selection_forward_stepwise(criterion="R2 (out-of-sample)", is_plot=is_plot)
        self.feature_selection_ridge_lasso(is_plot=is_plot)
        self.feature_selection_least_angle_regression(is_plot=is_plot)

        if is_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_best_subset_summary[i][4] for i in range(1, self.p+1)], "-o", label="best subset")
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][4] for i in range(1, self.p+1)], "-o", label="forward stepwise")
            #plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_ridge_summary[i][1] for i in range(1, self.p+1)], "-o", label="ridge")
            plt.plot([i for i in np.arange(1, self.p+1, 1) if i in self.feature_selection_lasso_summary.keys()], [self.feature_selection_lasso_summary[i][1] for i in np.arange(1, self.p+1, 1) if i in self.feature_selection_lasso_summary.keys()], "-o", label="lasso")
            plt.plot([i for i in np.arange(1, self.p+1, 1) if i in self.feature_selection_least_angle_regression_summary.keys()], [self.feature_selection_least_angle_regression_summary[i][1] for i in np.arange(1, self.p+1, 1) if i in self.feature_selection_least_angle_regression_summary.keys()], "-o", label="least angle regression")
            plt.hlines(y=self.feature_selection_best_subset_summary[0][4], xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1],
                       color='gray', linestyle='--', label="no feature")
            plt.xlabel("number of features")
            plt.ylabel("R^2 (out-of-sample)")
            plt.title("Feature selection")
            plt.legend()

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 2, 1)
            ar = np.zeros((self.p, self.p)); ar[:] = np.nan
            for feature_num in range(1, self.p+1, 1):
                selected_feature_idx = self.feature_selection_best_subset_summary[feature_num][0]
                ar[feature_num-1, selected_feature_idx] = 1
            plt.imshow(ar, cmap='Blues', vmin=0, vmax=1, alpha=0.5, aspect='auto')
            for i in range(self.p):
                plt.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5)
                plt.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=0.5)
            plt.xticks(range(self.p), self.X_columns, rotation=45)
            plt.yticks(range(self.p), range(1, self.p+1))
            plt.xlabel("selected feature")
            plt.ylabel("feature number")
            plt.title("Best subset selection")

            plt.subplot(2, 2, 2)        
            ar = np.zeros((self.p, self.p)); ar[:] = np.nan
            for feature_num in range(1, self.p+1, 1):
                selected_feature_idx = self.feature_selection_forward_stepwise_summary[feature_num][0]
                ar[feature_num-1, selected_feature_idx] = 1
            plt.imshow(ar, cmap='Blues', vmin=0, vmax=1, alpha=0.5, aspect='auto')
            for i in range(self.p):
                plt.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5)
                plt.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=0.5)
            plt.xticks(range(self.p), self.X_columns, rotation=45)
            plt.yticks(range(self.p), range(1, self.p+1))
            plt.xlabel("selected feature")
            plt.ylabel("feature number")
            plt.title("Forward stepwise selection")

            plt.subplot(2, 2, 3)
            ar = np.zeros((self.p, self.p)); ar[:] = np.nan
            for feature_num in range(1, self.p+1, 1):
                if feature_num in self.feature_selection_lasso_summary.keys():
                    selected_feature_idx = self.feature_selection_lasso_summary[feature_num][0]
                    ar[feature_num-1, selected_feature_idx] = 1
            plt.imshow(ar, cmap='Blues', vmin=0, vmax=1, alpha=0.5, aspect='auto')
            for i in range(self.p):
                plt.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5)
                plt.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=0.5)
            plt.xticks(range(self.p), self.X_columns, rotation=45)
            plt.yticks(range(self.p), range(1, self.p+1))
            plt.xlabel("selected feature")
            plt.ylabel("feature number")
            plt.title("Lasso selection")

            plt.subplot(2, 2, 4)
            ar = np.zeros((self.p, self.p)); ar[:] = np.nan
            for feature_num in range(1, self.p+1, 1):
                if feature_num in self.feature_selection_least_angle_regression_summary.keys():
                    selected_feature_idx = self.feature_selection_least_angle_regression_summary[feature_num][0]
                    ar[feature_num-1, selected_feature_idx] = 1
            plt.imshow(ar, cmap='Blues', vmin=0, vmax=1, alpha=0.5, aspect='auto')
            for i in range(self.p):
                plt.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5)
                plt.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=0.5)
            plt.xticks(range(self.p), self.X_columns, rotation=45)
            plt.yticks(range(self.p), range(1, self.p+1))
            plt.xlabel("selected feature")
            plt.ylabel("feature number")
            plt.title("Least angle regression")

            plt.suptitle("feature selection")
            plt.tight_layout()

    def feature_selection_best_subset(self, criterion="R2 (out-of-sample)", is_plot=True):
        '''
        Feature selection by best subset selection.
        params:
            criterion: str, "R2 (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "R2 (out-of-sample)"
        
        return:
            self.feature_selection_best_subset_summary: dict
                "feature_number": [feature_idx, R^2 (in-sample), AIC (in-sample), BIC (in-sample), R^2 (out-of-sample)]
        '''
        print("--- Feature selection by best subset ---")
        self.feature_selection_best_subset_summary = collections.defaultdict(list)
        self.feature_selection_best_subset_summary["feature_number"] = ["feature_idx", "R^2 (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "R^2 (out-of-sample)"]
        
        log = collections.defaultdict(list)
        ols = sm.OLS(self.Y, sm.add_constant(self.X, prepend=True)[:, 0].reshape((-1, 1))).fit()
        ols_oos = sm.OLS(self.Y_train, sm.add_constant(self.X_train, prepend=True)[:, 0].reshape((-1, 1))).fit()
        Y_pred = ols_oos.predict(sm.add_constant(self.X_test, prepend=True)[:, 0].reshape((-1, 1))).reshape(-1, 1)
        R2_oos = 1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2))
        self.feature_selection_best_subset_summary[0] = [[0], ols.rsquared, ols.aic, ols.bic, R2_oos]
        log[0].append([[], ols.rsquared, ols.aic, ols.bic, R2_oos])

        for feature_num in range(1, self.p + 1):
            for feature_idx in itertools.combinations(range(1, self.p+1), feature_num):
                feature_idx = [0] + list(feature_idx)
                ols = sm.OLS(self.Y, sm.add_constant(self.X, prepend=True)[:, feature_idx]).fit()
                ols_oos = sm.OLS(self.Y_train, sm.add_constant(self.X_train, prepend=True)[:, feature_idx]).fit()
                Y_pred = ols_oos.predict(sm.add_constant(self.X_test, prepend=True)[:, feature_idx]).reshape(-1, 1)
                R2_oos = 1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2))
                log[feature_num].append([list(np.array(feature_idx[1:])-1), ols.rsquared, ols.aic, ols.bic, R2_oos])

        plt.figure(figsize=(8, 6))
        plt.subplot(4, 1, 1)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[1])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[1] for i in log[feature_num]], "o", color="gray")
            if criterion == "R2 (in-sample)":
                print("feature_num: %d, best feature: %s, R^2 (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, R^2 (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][1] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel(r"$R^2$ (in-sample)")

        plt.subplot(4, 1, 2)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[2])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[2] for i in log[feature_num]], "o", color="gray")
            if criterion == "AIC (in-sample)":
                print("feature_num: %d, best feature: %s, R^2 (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, R^2 (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][2] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("AIC (in-sample)")

        plt.subplot(4, 1, 3)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[3])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[3] for i in log[feature_num]], "o", color="gray")
            if criterion == "BIC (in-sample)":
                print("feature_num: %d, best feature: %s, R^2 (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, R^2 (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][3] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("BIC (in-sample)")

        plt.subplot(4, 1, 4)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[4])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[4] for i in log[feature_num]], "o", color="gray")
            if criterion == "R2 (out-of-sample)":
                print("feature_num: %d, best feature: %s, R^2 (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, R^2 (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][4] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel(r"$R^2$ (out-of-sample)")
        plt.suptitle("Feature selection by best subset selection with criterion: %s" % criterion)
        plt.tight_layout()

        if not is_plot:
            plt.close(plt.gcf())

        return self.feature_selection_best_subset_summary

    def feature_selection_forward_stepwise(self, criterion="R2 (out-of-sample)", is_plot=True):
        '''
        Feature selection by forward stepwise selection.
        params:
            criterion: str, "R2 (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "R2 (out-of-sample)"
        
        return:
            self.feature_selection_forward_stepwise_summary: dict
                "feature_number": [feature_idx, R^2 (in-sample), AIC (in-sample), BIC (in-sample), R^2 (out-of-sample)]
        '''

        print("--- Feature selection by forward stepwise selection ---")
        self.feature_selection_forward_stepwise_summary = collections.defaultdict(list)
        self.feature_selection_forward_stepwise_summary["feature_number"] = ["feature_idx", "R^2 (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "R^2 (out-of-sample)"]

        selected_feature = set()
        log = collections.defaultdict(list)
        for feature_num in np.arange(1, self.p+1, 1):
            for feature_idx in range(self.p):
                if feature_idx not in selected_feature:
                    ols = sm.OLS(self.Y, sm.add_constant(self.X[:, list(selected_feature) + [feature_idx]], prepend=True)).fit()
                    ols_oos = sm.OLS(self.Y_train, sm.add_constant(self.X_train[:, list(selected_feature) + [feature_idx]], prepend=True)).fit()
                    Y_pred = ols_oos.predict(sm.add_constant(self.X_test[:, list(selected_feature) + [feature_idx]], prepend=True)).reshape(-1, 1)
                    R2_oos = 1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2))
                    log[feature_num].append([list(selected_feature) + [feature_idx], ols.rsquared, ols.aic, ols.bic, R2_oos])
            if criterion == "R2 (in-sample)":
                log[feature_num].sort(reverse=True, key=lambda x: x[1])
            if criterion == "AIC (in-sample)":
                log[feature_num].sort(reverse=False, key=lambda x: x[2])
            if criterion == "BIC (in-sample)":
                log[feature_num].sort(reverse=False, key=lambda x: x[3])
            if criterion == "R2 (out-of-sample)":
                log[feature_num].sort(reverse=True, key=lambda x: x[4])
            selected_feature.add(log[feature_num][0][0][-1])
            print("feature_num: %d, best feature: %s, R^2 (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, R^2 (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
            self.feature_selection_forward_stepwise_summary[feature_num] = log[feature_num][0]

        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(4,1,1)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][1] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("R^2 (in-sample)")
            plt.subplot(4,1,2)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][2] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("AIC (in-sample)")
            plt.subplot(4,1,3)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][3] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("BIC (in-sample)")
            plt.subplot(4,1,4)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][4] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("R^2 (out-of-sample)")
            plt.suptitle("Feature selection by forward stepwise selection with criterion: %s" % criterion)
            plt.tight_layout()

        return self.feature_selection_forward_stepwise_summary

    def feature_selection_ridge_lasso(self, beta_threshold = 1e-3, is_plot=True):
        '''
        Feature selection by ridge regression and lasso regression.
        Feature selection is based on the absolute value of the coefficients.
        Feature with absolute value of coefficient greater than beta_threshold is selected.
        params:
            beta_threshold: float, threshold for feature selection
        return:
            self.feature_selection_ridge_summary: dict
                "feature_number": [feature_idx, R^2 (out-of-sample)]
        '''
        # Ridge regression
        print('--- Feature selection by ridge regression ---')
        self.feature_selection_ridge_summary = collections.defaultdict(list)
        self.feature_selection_ridge_summary["feature_number"] = ["feature_idx", "R^2 (out-of-sample)"]
        alpha_list = np.logspace(-6, 6, num=10000, base=10)
        log = collections.defaultdict(list)

        for alpha in alpha_list:
            ridge = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(self.X_train, self.Y_train)
            Y_pred = ridge.predict(self.X_test).reshape(-1, 1)
            R2_oos = 1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2))
            log[alpha] = [ridge.coef_.T, ridge.intercept_, R2_oos]

        critical_alpha = []
        for i in range(len(alpha_list)-1, 0, -1):
            alpha = alpha_list[i]
            coef_1 = log[alpha_list[i]][0].flatten(); coef_2 = log[alpha_list[i-1]][0].flatten()
            coef_1_nonzero_ct = len(np.where(np.abs(coef_1) > beta_threshold)[0])
            coef_2_nonzero_ct = len(np.where(np.abs(coef_2) > beta_threshold)[0])
            if coef_2_nonzero_ct > coef_1_nonzero_ct:
                self.feature_selection_ridge_summary[coef_1_nonzero_ct] = [list(np.where(np.abs(coef_1) > beta_threshold)[0]), log[alpha][2]]
                print("Ridge feature number: %d, alpha: %.4f, selected feature index: %s, R^2 (out-of-sample): %.4f" % (coef_1_nonzero_ct, alpha, self.feature_selection_ridge_summary[coef_1_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)
            if i == 1:
                self.feature_selection_ridge_summary[coef_2_nonzero_ct] = [list(np.where(np.abs(coef_2) > beta_threshold)[0]), log[alpha][2]]
                print("Ridge feature number: %d, alpha: %.4f, selected feature index: %s, R^2 (out-of-sample): %.4f" % (coef_2_nonzero_ct, alpha, self.feature_selection_ridge_summary[coef_2_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)

        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(2,1,1)
            plt.plot(alpha_list, [log[i][2] for i in alpha_list])
            plt.xscale("log"); plt.xlabel("alpha"); plt.ylabel(r"$R^2$ (out-of-sample)")
            plt.subplot(2,1,2)
            for feature_idx in range(self.p):
                plt.plot(alpha_list, [log[i][0][feature_idx, 0] for i in alpha_list],  label=self.X_columns[feature_idx])
            plt.axhline(y=0, color='black', linestyle='--')
            plt.vlines(critical_alpha, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label="critical alpha")
            plt.xscale("log"); plt.xlabel("alpha"); plt.ylabel(r"$\beta$"); plt.legend()
            plt.suptitle("Feature selection by ridge regression")
            plt.tight_layout()

        # Lasso regression
        print('--- Feature selection by lasso regression ---')
        self.feature_selection_lasso_summary = collections.defaultdict(list)
        self.feature_selection_lasso_summary["feature_number"] = ["feature_idx", "R^2 (out-of-sample)"]
        alpha_list = np.logspace(-6, 6, num=10000, base=10)
        log = collections.defaultdict(list)

        for alpha in alpha_list:
            lasso = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=True)
            lasso.fit(self.X_train, self.Y_train)
            Y_pred = lasso.predict(self.X_test).reshape(-1, 1)
            R2_oos = 1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2))
            log[alpha] = [lasso.coef_.T.reshape((-1, 1)), lasso.intercept_, R2_oos]

        critical_alpha = []
        for i in range(len(alpha_list)-1, 0, -1):
            alpha = alpha_list[i]
            coef_1 = log[alpha_list[i]][0].flatten(); coef_2 = log[alpha_list[i-1]][0].flatten()
            coef_1_nonzero_ct = len(np.where(np.abs(coef_1) > beta_threshold)[0])
            coef_2_nonzero_ct = len(np.where(np.abs(coef_2) > beta_threshold)[0])
            if coef_2_nonzero_ct > coef_1_nonzero_ct:
                self.feature_selection_lasso_summary[coef_1_nonzero_ct] = [list(np.where(np.abs(coef_1) > beta_threshold)[0]), log[alpha][2]]
                print("Lasso feature number: %d, alpha: %.4f, selected feature index: %s, R^2 (out-of-sample): %.4f" % (coef_1_nonzero_ct, alpha, self.feature_selection_lasso_summary[coef_1_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)
            if i == 1:
                self.feature_selection_lasso_summary[coef_2_nonzero_ct] = [list(np.where(np.abs(coef_2) > beta_threshold)[0]), log[alpha][2]]
                print("Lasso feature number: %d, alpha: %.4f, selected feature index: %s, R^2 (out-of-sample): %.4f" % (coef_2_nonzero_ct, alpha, self.feature_selection_lasso_summary[coef_2_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)

        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(2,1,1)
            plt.plot(alpha_list, [log[i][2] for i in alpha_list])
            plt.xscale("log"); plt.xlabel("alpha"); plt.ylabel(r"$R^2$ (out-of-sample)")
            plt.subplot(2,1,2)
            for feature_idx in range(self.p):
                plt.plot(alpha_list, [log[i][0][feature_idx, 0] for i in alpha_list],  label=self.X_columns[feature_idx])
            plt.axhline(y=0, color='black', linestyle='--')
            plt.vlines(critical_alpha, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label="critical alpha")
            plt.xscale("log"); plt.xlabel("alpha"); plt.ylabel(r"$\beta$"); plt.legend()
            plt.suptitle("Feature selection by lasso regression")
            plt.tight_layout()

    def feature_selection_least_angle_regression(self, beta_threshold = 1e-3, is_plot=True):
        '''
        Feature selection by least angle regression.
        Feature selection is based on the absolute value of the coefficients.
        Feature with absolute value of coefficient greater than beta_threshold is selected.
        params:
            beta_threshold: float, threshold for feature selection
        return:
            self.feature_selection_least_angle_regression_summary: dict
                "feature_number": [feature_idx, R^2 (out-of-sample)]
        '''
        print('--- Least Angle Regression ---')
        self.feature_selection_least_angle_regression_summary = collections.defaultdict(list)
        self.feature_selection_least_angle_regression_summary["feature_number"] = ["feature_idx"]
        lars = sklearn.linear_model.Lars(n_nonzero_coefs=self.p, fit_intercept=True)
        lars.fit(self.X, self.Y)
        alpha_path = lars.alphas_
        coef_path = lars.coef_path_.T
        R2_oos = []
        interpolator = scipy.interpolate.interp1d(alpha_path, coef_path, axis=0, fill_value="extrapolate")
        for alpha_ in alpha_path:
            coef = interpolator(alpha_).flatten()
            Y_pred = np.dot(self.X_test, coef.reshape(-1, 1)) + lars.intercept_
            R2_oos.append(1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2)))

        feature_num = 0; critical_alpha = []
        for i in range(len(alpha_path)):
            alpha = alpha_path[i]
            coef = coef_path[i, :]
            if len(np.where(np.abs(coef) > beta_threshold)[0]) > feature_num:
                feature_num = len(np.where(np.abs(coef) > beta_threshold)[0])
                self.feature_selection_least_angle_regression_summary[feature_num] = [list(np.where(np.abs(coef) > beta_threshold)[0]), R2_oos[i]]
                print("Least Angle Regression: feature number: %d, alpha: %.4f, selected feature index: %s, R^2 (out-of-sample): %.4f" % (feature_num, alpha, self.feature_selection_least_angle_regression_summary[feature_num][0], R2_oos[i]))
                critical_alpha.append(alpha)

        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(2,1,1)
            for i in range(coef_path.shape[1]):
                plt.plot(alpha_path, coef_path[:, i], "-o", label=self.X_columns[i])
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xlabel("alpha"); plt.ylabel(r"$\beta$"); plt.legend()

            plt.subplot(2,1,2)
            plt.plot(alpha_path, R2_oos, "-o")
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xlabel("alpha"); plt.ylabel(r"$R^2$ (out-of-sample)")
            plt.suptitle("least angle regression")
            plt.tight_layout()

    def _smoother(self, x, y, type="polynomial"):
        '''
        Smoother.
        params:
            x: np.ndarray, shape (n_samples, n_features), input data
            y: np.ndarray, shape (n_samples, 1), target data
            type: str, type of smoother, "polynomial", "spline", "loess", "kernel"

        return:
            fit_x: np.ndarray, shape (n_samples, 1), x values for fitted line
            fit_y: np.ndarray, shape (n_samples, 1), y values for fitted line
        '''
        x = x.flatten(); y = y.flatten()
        if type == "polynomial":
            x_temp = np.concatenate([x.reshape((-1, 1)), np.power(x.reshape((-1, 1)), 2)], axis=1)
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(x_temp, y)
            R2 = linreg.score(x_temp, y)
            fit_x = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
            fit_y = linreg.predict(np.concatenate([fit_x, np.power(fit_x, 2)], axis=1)).reshape(-1, 1)
            return {"fit_x": fit_x, "fit_y": fit_y, "R2": R2}

        if type == "spline":
            sort_idx = np.argsort(x)
            spline = scipy.interpolate.CubicSpline(x[sort_idx].flatten(), y[sort_idx].flatten())
            fit_x = np.linspace(x.min(), x.max(), 1000)
            fit_y = spline(fit_x)
            return {"fit_x": fit_x.reshape((-1, 1)), "fit_y": fit_y.reshape((-1, 1)), "R2": np.nan}
        
        if type == "loess":
            loess = statsmodels.nonparametric.smoothers_lowess.lowess(y, x, frac=0.3, it=0)
            fit_x = loess[:, 0].reshape((-1, 1))
            fit_y = loess[:, 1].reshape((-1, 1))
            return {"fit_x": fit_x, "fit_y": fit_y, "R2": np.nan}
        
        if type == "kernel":
            x = x.reshape(-1, 1)
            kernel = statsmodels.nonparametric.kernel_regression.KernelReg(endog=y, exog=x, var_type="c", bw="cv_ls")
            fit_x = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
            fit_y = kernel.fit(fit_x)[0].reshape(-1, 1)
            return {"fit_x": fit_x, "fit_y": fit_y, "R2": np.nan}

        raise Exception("Unknown type for smoother: %s" % type)

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/prostate_cancer.csv"), index_col=0)
data = data[data["train"] == "T"]
data = data[["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45", "lpsa"]]
column_name = data.columns[:-1].tolist()
X = data.iloc[:, 0:(data.shape[1]-1)].to_numpy()
Y = data.iloc[:, -1].to_numpy().reshape(-1, 1)
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/prostate_cancer.csv"), index_col=0)
data = data[data["train"] == "F"]
data = data[["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45", "lpsa"]]
column_name = data.columns[:-1].tolist()
X_test = data.iloc[:, 0:(data.shape[1]-1)].to_numpy()
Y_test = data.iloc[:, -1].to_numpy().reshape(-1, 1)

#model = linear_regression(X, Y, X_test=X_test, Y_test=Y_test, X_columns=column_name, is_normalize=True, test_size_ratio=0.2)
#model.visualize_data()
#model.fit(is_output=False)
#outlier_idx = model.outlier(threshold="strict", is_output=True)["outlier_idx"]
#X_new = X[~np.isin(np.arange(X.shape[0]), outlier_idx), :]
#Y_new = Y[~np.isin(np.arange(Y.shape[0]), outlier_idx), :]

#model = linear_regression(X_new, Y_new, X_test=X_test, Y_test=Y_test, X_columns=column_name, is_normalize=True, test_size_ratio=0.2)
#model.fit(is_output=False)
#model.visualize_data()
#model.colinearity()
#model.homoscedasticity()
#model.test_residual_normality_independence()
#model.feature_target_correlation()
#model.feature_selection_all()

# %%
class ridge_lasso_regression(linear_regression):
    def __init__(self, X, Y, X_test, Y_test, X_columns=None, is_normalize=True, test_size_ratio=0.2, regularization="ridge"):
        super().__init__(X, Y, X_test=X_test, Y_test=Y_test, X_columns=X_columns, is_normalize=is_normalize, test_size_ratio=test_size_ratio)
        self.regularization = regularization

    def fit(self, alpha=None, is_output=True):
        if not alpha:
            alpha = self.optimal_alpha(is_plot=False)
            self.alpha = alpha
        if self.regularization == "ridge":
            self.model = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=True)
            self.model.fit(self.X_train, self.Y_train)
        if self.regularization == "lasso":
            self.model = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=True)
            self.model.fit(self.X_train, self.Y_train)

        if is_output:
            plt.figure(figsize=(6, 4))
            plt.plot(self.X_columns, self.model.coef_.flatten(), "o-", label="coef")
            plt.axhline(y=0, color='black', linestyle='--')
            plt.legend(title=r"$\alpha=$"+"%.3f" % alpha)
            plt.title("{} regression".format(self.regularization))
            plt.xlabel("feature"); plt.ylabel(r"$\beta$")

    def predict(self, X):
        if self.is_normalize:
            X = (X - self.X_mean) / self.X_std
        Y_pred = self.model.predict(X).reshape(-1, 1)
        if self.is_normalize:
            Y_pred = Y_pred * self.Y_std + self.Y_mean
        return Y_pred

    def optimal_alpha(self, is_plot=True):
        alpha_list = np.logspace(-6, 6, num=1000, base=10)
        R2_oss_hist = []; beta_hist = []
        for alpha in alpha_list:
            if self.regularization == "ridge":
                ridge = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(self.X_train, self.Y_train)
                Y_pred = ridge.predict(self.X_test).reshape(-1, 1)
                beta_hist.append(list(ridge.coef_.flatten()))
            elif self.regularization == "lasso":
                lasso = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=True)
                lasso.fit(self.X_train, self.Y_train)
                Y_pred = lasso.predict(self.X_test).reshape(-1, 1)
                beta_hist.append(list(lasso.coef_.flatten()))
            R2_oos = 1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2))
            R2_oss_hist.append(R2_oos)
        beta_hist = np.array(beta_hist)

        if is_plot:
            plt.figure(figsize=(6, 6))
            plt.subplot(2, 1, 1)
            plt.plot(alpha_list, R2_oss_hist)
            plt.vlines(alpha_list[np.argmax(R2_oss_hist)], ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label=r"$\alpha^*$"+"=%.3f" % alpha_list[np.argmax(R2_oss_hist)])
            plt.xscale("log")
            plt.xlabel(r"$\alpha$"); plt.ylabel(r"$R^2$ (out-of-sample)")
            plt.subplot(2, 1, 2)
            for i in range(beta_hist.shape[1]):
                plt.plot(alpha_list, beta_hist[:, i], label=self.X_columns[i])
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xscale("log")
            plt.legend()
            plt.xlabel(r"$\alpha$"); plt.ylabel(r"$\beta$")
            plt.suptitle("{} regression (out-of-sample R2)".format(self.regularization))
            plt.tight_layout()

        return alpha_list[np.argmax(R2_oss_hist)]

#model = ridge_lasso_regression(X_new, Y_new, X_test, Y_test, X_columns=column_name, regularization="lasso")
#model.optimal_alpha()
#model.fit()

#%%
class principal_component_regression(linear_regression):
    def __init__(self, X, Y, X_test=None, Y_test=None, X_columns=None, is_normalize=True, test_size_ratio=0.2):
        super().__init__(X, Y, X_test=X_test, Y_test=Y_test, X_columns=X_columns, is_normalize=is_normalize, test_size_ratio=test_size_ratio)

    def fit(self, factor_num=None, is_plot=True):
        if not factor_num:
            factor_num = self.optimal_factor_number(criterion="R^2 (out-of-sample)", is_plot=False)
        U, S, VT = np.linalg.svd(self.X, full_matrices=False, compute_uv=True)
        Z = np.dot(self.X, VT.T[:, 0:factor_num])
        self.ols = sm.OLS(self.Y, sm.add_constant(Z, prepend=True)).fit()
        self.intercept = self.ols.params[0]
        self.beta = VT.T[:, 0:factor_num].dot(self.ols.params[1:]).reshape(-1, 1)
        self.V = VT.T[:, 0:factor_num]

        if is_plot:
            plt.figure(figsize=(6, 4))
            plt.subplot(2, 1, 1)
            plt.plot(["pc {}".format(i) for i in range(factor_num)], self.ols.params[1:], "o-")
            plt.axhline(y=0, color='black', linestyle='--')
            plt.ylabel(r"$\beta$"); plt.xticks(rotation=0)
            plt.subplot(2, 1, 2)
            plt.plot(self.X_columns, self.beta, "o-", label="coef")
            plt.axhline(y=0, color='black', linestyle='--')
            plt.ylabel(r"$\beta$"); plt.xticks(rotation=0)
            plt.suptitle("Principal component regression")
            plt.tight_layout()

            n_cols = 2; n_rows = int(np.ceil(factor_num / n_cols))                
            plt.figure(figsize=(4*n_cols, 2*n_rows))
            for i in range(factor_num):
                plt.subplot(n_rows, n_cols, i+1)
                plt.plot(self.X_columns, self.V[:, i], "o-", label=r"pc {}".format(i))
                plt.axhline(y=0, color='black', linestyle='--')
                plt.xticks(rotation=45)
                plt.ylabel(r"$v_{}$".format(i))
                plt.legend()
                plt.suptitle(r"$\{v_i\}_{i=1}^K$: principal components")
                plt.tight_layout()

    def predict(self, X):
        if self.is_normalize:
            X = (X - self.X_mean) / self.X_std
        Z_pred = np.dot(X, self.V)
        Y_pred = self.ols.predict(sm.add_constant(Z_pred, prepend=True)).reshape(-1, 1)
        if self.is_normalize:
            Y_pred = Y_pred * self.Y_std + self.Y_mean
        return Y_pred

    def optimal_factor_number(self, criterion = "R^2 (out-of-sample)", is_plot=True):
        
        U, S, VT = np.linalg.svd(self.X, full_matrices=False, compute_uv=True)
        Z = np.dot(self.X, VT.T).copy()

        U_train, S_train, VT_train = np.linalg.svd(self.X_train, full_matrices=False, compute_uv=True)
        Z_train = np.dot(U_train, np.diag(S_train)).copy()
        Z_test = np.dot(self.X_test, VT_train.T).copy()

        # Eigenvalue spectrum
        # Marchenko-Pastur distribution
        marchenko_pastur_bound = [(1-np.sqrt(self.p/self.n))**2, (1+np.sqrt(self.p/self.n))**2]
        # Tracy-Widom distribution
        tracy_widom_bound = [0, np.power(1+np.sqrt(self.p/self.n), 2)+1.27*(np.sqrt(self.p)+np.sqrt(self.n))*np.power(1/np.sqrt(self.p)+1/np.sqrt(self.n), 1/3)/self.n]
        # information criterion
        ic = [np.log(np.sum(S[0:i]))+i*(self.n+self.p)*np.log(self.n*self.p/(self.n+self.p)) / (self.n*self.p) for i in np.arange(1, self.p+1, 1)]
        ic_arg = np.argmin(ic) + 1
        if is_plot:
            plt.figure(figsize=(6, 3))
            plt.plot(np.arange(1, self.p+1, 1), S_train, "o-")
            plt.hlines(y=marchenko_pastur_bound[1], xmin=0, xmax=self.p-1, color="red", linestyle="--", label="Marchenko-Pastur")
            plt.hlines(y=tracy_widom_bound[1], xmin=0, xmax=self.p-1, color="blue", linestyle="--", label="Tracy-Widom")
            plt.axvline(x=ic_arg, color="green", linestyle="--", label="information criterion")
            plt.xlabel("factor number"); plt.ylabel("eigenvalue")
            plt.title("Eigenvalue spectrum")
            plt.legend()
            plt.tight_layout()

        log = collections.defaultdict(list)
        log["feature_number"] = ["R^2 (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "R^2 (out-of-sample)"]
        for factor_num in np.arange(1, self.p+1, 1):
            ols = sm.OLS(self.Y, sm.add_constant(Z[:, 0:factor_num], prepend=True)).fit()
            ols_oos = sm.OLS(self.Y_train, sm.add_constant(Z_train[:, 0:factor_num], prepend=True)).fit()
            Y_pred = ols_oos.predict(sm.add_constant(Z_test[:, 0:factor_num], prepend=True)).reshape(-1, 1)
            R2_oos = 1 - np.sum((self.Y_test - Y_pred)**2)/(np.sum((self.Y_test - np.mean(self.Y_test))**2))
            log[factor_num] = [ols.rsquared, ols.aic, ols.bic, R2_oos]

        plt.figure(figsize=(8, 8))
        plt.subplot(4, 1, 1)
        plt.plot(np.arange(1, self.p+1, 1), [log[i][0] for i in np.arange(1, self.p+1, 1)], "o-")
        if criterion == "R^2 (in-sample)":
            optimal_factor_num = np.argmax([log[i][0] for i in np.arange(1, self.p+1, 1)])
            plt.axvline(x=optimal_factor_num, color="red", linestyle="--", label="optimal factor number")
        plt.xlabel("factor number"); plt.ylabel(r"$R^2$ (in-sample)")
        plt.subplot(4, 1, 2)
        plt.plot(np.arange(1, self.p+1, 1), [log[i][1] for i in np.arange(1, self.p+1, 1)], "o-")
        if criterion == "AIC (in-sample)":
            optimal_factor_num = np.argmin([log[i][1] for i in np.arange(1, self.p+1, 1)]) + 1
            plt.axvline(x=optimal_factor_num, color="red", linestyle="--", label="optimal factor number")
        plt.xlabel("factor number"); plt.ylabel("AIC (in-sample)")
        plt.subplot(4, 1, 3)
        plt.plot(np.arange(1, self.p+1, 1), [log[i][2] for i in np.arange(1, self.p+1, 1)], "o-")
        if criterion == "BIC (in-sample)":
            optimal_factor_num = np.argmin([log[i][2] for i in np.arange(1, self.p+1, 1)]) + 1
            plt.axvline(x=optimal_factor_num, color="red", linestyle="--", label="optimal factor number")
        plt.xlabel("factor number"); plt.ylabel("BIC (in-sample)")
        plt.subplot(4, 1, 4)
        plt.plot(np.arange(1, self.p+1, 1), [log[i][3] for i in np.arange(1, self.p+1, 1)], "o-")
        if criterion == "R^2 (out-of-sample)":
            optimal_factor_num = np.argmax([log[i][3] for i in np.arange(1, self.p+1, 1)]) + 1
            plt.axvline(x=optimal_factor_num, color="red", linestyle="--", label="optimal factor number")
        plt.xlabel("factor number"); plt.ylabel(r"$R^2$ (out-of-sample)")
        plt.suptitle("Principal component regression with criterion: %s" % criterion)
        plt.tight_layout()
        return optimal_factor_num

#model = principal_component_regression(X_new, Y_new, X_test=X_test, Y_test=Y_test, X_columns=column_name, is_normalize=True, test_size_ratio=0.2)
#model.optimal_factor_number()
#model.fit(factor_num=5, is_plot=True)

#%%
class partial_least_square_regression(linear_regression):
    def __init__(self, X, Y, X_test=None, Y_test=None, X_columns=None, is_normalize=True, test_size_ratio=0.2):
        super().__init__(X, Y, X_test=X_test, Y_test=Y_test, X_columns=X_columns, is_normalize=is_normalize, test_size_ratio=test_size_ratio)

    def fit(self, factor_num=None, is_output=True):
        if not factor_num:
            factor_num = self.optimal_factor_number(criterion="R^2 (out-of-sample)", is_plot=False)
            self.factor_number = factor_num
        self.pls = sklearn.cross_decomposition.PLSRegression(n_components=factor_num, scale=True)
        self.pls.fit(self.X, self.Y)

        if is_output:
            plt.figure(figsize=(6, 4))
            plt.plot(self.X_columns, self.pls.coef_.flatten(), "o-", label="coef")
            plt.axhline(y=0, color='black', linestyle='--')
            plt.legend(title="factor number: %d" % factor_num)
            plt.title("Partial least square regression")
            plt.xlabel("feature"); plt.ylabel(r"$\beta$")
            plt.xticks(rotation=0)

    def predict(self, X):
        if self.is_normalize:
            X = (X - self.X_mean) / self.X_std
        Y_pred = self.pls.predict(X).reshape(-1, 1)
        if self.is_normalize:
            Y_pred = Y_pred * self.Y_std + self.Y_mean
        return Y_pred

    def optimal_factor_number(self, criterion = "R^2 (out-of-sample)", is_plot=True):
        log = []
        for factor_num in np.arange(1, self.p+1, 1):
            pls = sklearn.cross_decomposition.PLSRegression(n_components=factor_num, scale=True)
            pls.fit(self.X, self.Y)
            R2 = pls.score(self.X, self.Y)
            pls_oos = sklearn.cross_decomposition.PLSRegression(n_components=factor_num, scale=True)
            pls_oos.fit(self.X_train, self.Y_train)
            Y_pred = pls_oos.predict(self.X_test).reshape(-1, 1)
            R2_oos = pls_oos.score(self.X_test, self.Y_test)
            log.append([factor_num, R2, R2_oos])

        log = np.array(log)
        if is_plot:
            plt.figure(figsize=(6, 4))
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(1, self.p+1, 1), log[:, 1], "o-")
            if criterion == "R^2 (in-sample)":
                optimal_factor_num = np.argmax(log[:, 1]) + 1
                plt.axvline(x=optimal_factor_num, color="red", linestyle="--", label="optimal factor number")
            plt.xlabel("factor number"); plt.ylabel(r"$R^2$ (in-sample)")
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(1, self.p+1, 1), log[:, 2], "o-")
            if criterion == "R^2 (out-of-sample)":
                optimal_factor_num = np.argmax(log[:, 2]) + 1
                plt.axvline(x=optimal_factor_num, color="red", linestyle="--", label="optimal factor number")
            plt.xlabel("factor number"); plt.ylabel(r"$R^2$ (out-of-sample)")
            plt.suptitle("Partial least square regression with criterion: %s" % criterion)
            plt.tight_layout()

            return optimal_factor_num

#to be continued from here
#model = partial_least_square_regression(X_new, Y_new, X_test=X_test, Y_test=Y_test, X_columns=column_name, is_normalize=True, test_size_ratio=0.2)
#model.optimal_factor_number()
#model.fit(factor_num=5, is_output=True)

#%%
class linear_regression_Bspline(linear_regression):
    def __init__(self, X, Y, bs_col, bs_df, X_test=None, Y_test=None, X_columns=None, is_normalize=True, test_size_ratio=0.2):
        super().__init__(X, Y, X_test=X_test, Y_test=Y_test, X_columns=X_columns, is_normalize=is_normalize, test_size_ratio=test_size_ratio)

        self.bs_col = bs_col
        self.bs_df = bs_df
        if bs_df < 5:
            raise ValueError("B-spline degree of freedom less than 5.")
        self.knots_num = bs_df - 3 - 1 # for B-spline, df = knots_num + degree + 1
        self.knots = dict()

        self.X_columns_bs = []
        self.X_bs = np.zeros((self.n, 0))
        self.X_train_bs = np.zeros((self.X_train.shape[0], 0))
        self.X_test_bs = np.zeros((self.X_test.shape[0], 0))
        for j in range(self.p):
            if j in self.bs_col:
                self.X_columns_bs.extend(["{}_bs{}".format(self.X_columns[j], i) for i in range(self.bs_df)])

                knots = np.linspace(np.min(self.X[:, j]), np.max(self.X[:, j]), self.knots_num + 2)[1:(self.knots_num + 1)]
                formula = "bs(X, knots=({}), degree=3, include_intercept=False)".format(", ".join(["{:f}".format(k) for k in knots]))
                bs = patsy.dmatrix(formula, {"X": self.X[:, j]}, return_type='matrix')
                self.X_bs = np.hstack((self.X_bs, np.array(bs[:, 1:])))
                self.knots["all"] = knots
                del bs

                knots = np.linspace(np.min(self.X_train[:, j]), np.max(self.X_train[:, j]), self.knots_num + 2)[1:(self.knots_num + 1)]
                formula = "bs(X, knots=({}), degree=3, include_intercept=False)".format(", ".join(["{:f}".format(k) for k in knots]))
                bs = patsy.dmatrix(formula, {"X": self.X_train[:, j]}, return_type='matrix')
                self.X_train_bs = np.hstack((self.X_train_bs, np.array(bs[:, 1:])))
                self.knots["train"] = knots
                plt.legend()
                del bs

                knots = np.linspace(np.min(self.X_test[:, j]), np.max(self.X_test[:, j]), self.knots_num + 2)[1:(self.knots_num + 1)]
                formula = "bs(X, knots=({}), degree=3, include_intercept=False)".format(", ".join(["{:f}".format(k) for k in knots]))
                bs = patsy.dmatrix(formula, {"X": self.X_test[:, j]}, return_type='matrix')
                self.X_test_bs = np.hstack((self.X_test_bs, np.array(bs[:, 1:])))
                self.knots["test"] = knots
                del bs

            else:
                self.X_columns_bs.append(self.X_columns[j])
                self.X_bs = np.hstack((self.X_bs, self.X[:, j].reshape(-1, 1)))

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/prostate_cancer.csv"), index_col=0)
data = data[data["train"] == "T"]
data = data[["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45", "lpsa"]]
column_name = data.columns[:-1].tolist()
X = data.iloc[:, 0:(data.shape[1]-1)].to_numpy()
Y = data.iloc[:, -1].to_numpy().reshape(-1, 1)
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/prostate_cancer.csv"), index_col=0)
data = data[data["train"] == "F"]
data = data[["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45", "lpsa"]]
column_name = data.columns[:-1].tolist()
X_test = data.iloc[:, 0:(data.shape[1]-1)].to_numpy()
Y_test = data.iloc[:, -1].to_numpy().reshape(-1, 1)

model = linear_regression_Bspline(X, Y, bs_col=[0], bs_df=6, X_test=X_test, Y_test=Y_test, X_columns=column_name, is_normalize=True, test_size_ratio=0.2)
#model.visualize_data()
#model.fit(is_output=False)
#outlier_idx = model.outlier(threshold="strict", is_output=True)["outlier_idx"]
#X_new = X[~np.isin(np.arange(X.shape[0]), outlier_idx), :]
#Y_new = Y[~np.isin(np.arange(Y.shape[0]), outlier_idx), :]


#%%
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
import statsmodels.api as sm

# Data
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.2 * np.random.randn(100)

# Create natural cubic spline basis
#x_basis = dmatrix("cr(x, df=5)", {"x": x}, return_type='dataframe')
x_basis = dmatrix("bs(x, knots=(3, 6), degree=3, include_intercept=False)", {"x": x})
# Fit linear model on spline basis
model = sm.OLS(y, x_basis).fit()
# Predict
y_pred = model.predict(x_basis)

# Plot
'''
plt.scatter(x, y, alpha=0.4, label='Data')
plt.plot(x, y_pred, 'r', label='Natural Cubic Spline')
plt.legend()
plt.title("Natural Cubic Spline Regression")
plt.show()
'''

for i in range(x_basis.shape[1]):
    plt.plot(x, x_basis[:, i], label=f'Basis {i+1}')

#%%
print(x_basis.shape)
#np.array(x_basis)
x_basis


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Example knots (x values where spline pieces join)
x_knots = np.array([0, 1, 2.5, 4, 6, 8, 10])
y_knots = np.sin(x_knots)  # values at knots

# Natural cubic spline interpolation
spline = CubicSpline(x_knots, y_knots, bc_type='natural')

# Evaluate
x_dense = np.linspace(0, 10, 200)
y_dense = spline(x_dense)

# Plot
plt.plot(x_knots, y_knots, 'o', label='Knots')
plt.plot(x_dense, y_dense, '-', label='Natural cubic spline')
plt.legend()
plt.grid(True)
plt.title("Natural Cubic Spline with Custom Knots")
plt.show()

#%%
np.array()



# %%






