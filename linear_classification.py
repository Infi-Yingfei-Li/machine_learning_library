#%%
from cProfile import label
from nt import error, stat
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from tokenize import group
from numba.core.utils import benchmark
from scipy.integrate._ivp.radau import P
from seaborn import colors
import statsmodels.multivariate
import statsmodels.multivariate.manova
import statsmodels.stats.multivariate
import os, sys, copy, scipy, datetime, tqdm, collections, itertools
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
import sklearn.discriminant_analysis
import sklearn.utils.multiclass

import umap, pingouin

#%%
class linear_discriminant_analysis:
    def __init__(self, X, Y, X_test=None, Y_test=None, X_columns=None, is_normalize=True, test_size_ratio=0.2):
        '''
        Initialize the linear discriminant analysis model.
        params:
            X: feature matrix, shape (n, p)
            Y: target vector, shape (n, 1). Y is a vector encoding class information, with the first class as 0, the second class as 1, and so on.
            X_test: test feature matrix, shape (n_test, p)
            Y_test: test target vector, shape (n_test, 1)
                If X_test and Y_test are not provided, the data will be split into train and test sets based on the test_size_ratio.
            X_columns: feature names, list of length p. If None, use default names ["feature_0", "feature_1", ...]
            is_normalize: whether to normalize the data
            test_size_ratio: the ratio of test set size to the total size

        General workflow:
            (1) Fit LDA model;
            (2) Check normality within each class;
            (3) Check equal covariance matrices across classes;
            (4) Check multicollinearity of features;
            (5) Check outliers of the features;
            (6) Iteratively check each aspect above if remedies are applied;
            (7) Feature selection;
        '''
        self.X = X; self.Y = Y.reshape(-1, 1)
        self.X_columns = X_columns if X_columns is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.is_normalize = is_normalize
        self.test_size_ratio = test_size_ratio

        self.n = X.shape[0]; self.p = X.shape[1]; self.c = len(np.unique(Y.flatten()))

        if self.is_normalize:
            self.X_mean = np.mean(self.X, axis=0)
            self.X_std = np.std(self.X, axis=0, ddof=1)
            self.X = (self.X - self.X_mean) / self.X_std

        if (X_test is not None) and (Y_test is not None):
            self.X_train = self.X
            self.Y_train = self.Y
            self.X_test = X_test.copy()
            self.Y_test = Y_test.copy().reshape(-1, 1)
            if self.is_normalize:
                self.X_test = (X_test.copy() - self.X_mean) / self.X_std
        else:
            self.X_train = self.X.copy()[0:int(self.n*(1-self.test_size_ratio)), :]
            self.Y_train = self.Y.copy()[0:int(self.n*(1-self.test_size_ratio)), :]
            self.X_test = self.X.copy()[int(self.n*(1-self.test_size_ratio)):, :]
            self.Y_test = self.Y.copy()[int(self.n*(1-self.test_size_ratio)):, :]

    def fit(self, solver="eigen", is_output=True):
        '''
        Fit the linear discriminant analysis.
        params:
            solver: the solver to use. Options are "svd", "lsqr", "eigen". Default is "lsqr".
                "svd": singular value decomposition. Does not compute the covariance matrix, therefore is recommended for large datasets.
                "lsqr": least squares. Computes the covariance matrix. Can be combined with shrinkage or custom covariance matrix.
                "eigen": eigenvalue decomposition. Computes the covariance matrix and solves the eigenvalue problem. Can be used for small datasets with many features.
            is_output: whether to print the summary of the model
        '''
        alpha_opt = self.optimal_shrinkage_covariance()
        self.lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver=solver, shrinkage=alpha_opt, store_covariance=True, tol=1e-4)
        self.lda.fit(self.X, self.Y.flatten())

    def predict(self, X):
        '''
        Predict the target values for the given input features.
        params:
            X: feature matrix, shape (n, p)
        return:
            Y_pred: predicted target values, shape (n, 1)
        '''
        if not hasattr(self, "lda"):
            raise Exception("Fit the model first.")
        if self.is_normalize:
            X = (X - self.X_mean) / self.X_std
        Y_pred = self.lda.predict(X)
        return Y_pred

    def visualize_data(self):
        '''
        Visualize the data by:
            1. summary of the data
            2. plot each feature across observations (for time-series data)
            3. boxplot each feature across observations
            4. low-dimensional projection
        '''
        # describe the data
        df = pd.DataFrame(self.X, columns=self.X_columns)
        df["target"] = self.Y
        print("summary of the data:")
        print(df.describe())

        # plot each feature across observations
        ncol = 3; nrow = (self.p + 1) // ncol + 1
        plt.figure(figsize=(4*ncol, 3*nrow))
        for i in range(self.p):
            plt.subplot(nrow, ncol, i + 1)
            plt.plot(self.X[:, i])
            plt.ylabel(self.X_columns[i])
        plt.subplot(nrow, ncol, self.p + 1)
        plt.plot(self.Y, color="red")
        plt.ylabel("target")
        plt.suptitle("Feature across observations")
        plt.tight_layout()

        # boxplot each feature across observations
        nrow = 2; ncol = self.p//nrow + 1
        plt.figure(figsize=(1.5*ncol, 2*nrow))
        for i in range(self.p):
            plt.subplot(nrow, ncol, i + 1)
            plt.boxplot(self.X[:, i], widths=0.5, showfliers=True)
            plt.ylabel(self.X_columns[i])
        plt.subplot(nrow, ncol, self.p + 1)
        plt.boxplot(self.Y, widths=0.5, showfliers=True)
        plt.ylabel("target")
        plt.suptitle("Feature boxplot")
        plt.tight_layout()

        if self.p > 2:
            color_feature = self.Y.flatten()
            plt.figure(figsize=(12, 12))

            # projection by reduced rank linear discriminant analysis (RLDA)
            if self.c == 2:
                rrlda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='eigen', n_components=1, shrinkage='auto', store_covariance=True)
                X_lda = rrlda.fit_transform(self.X, self.Y.flatten())
                W_lda = rrlda.scalings_[:, 0].reshape(-1, 1)
                X_residual = self.X - self.X.dot(W_lda).dot(W_lda.T)
                U, S, V = np.linalg.svd(X_residual, full_matrices=False)
                W_pca = V[0, :].reshape(-1, 1)
                W = np.hstack((W_lda, W_pca))
                X_trans = self.X.dot(W)
                plt.subplot(2,2,1)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=1)
                plt.colorbar(label=r"$Y$")
                center = rrlda.means_.dot(W)
                plt.scatter(center[:, 0], center[:, 1], c=np.arange(self.c), cmap='RdBu_r', marker='x', s=500)
                for i in range(self.c):
                    plt.text(center[i, 0], center[i, 1], f"class {i}", fontsize=12, ha='center', va='center')
                
            if self.c > 2:
                rrlda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='eigen', n_components=2, shrinkage='auto', store_covariance=True)
                X_trans = rrlda.fit_transform(self.X, self.Y.flatten())
                W = rrlda.scalings_[:, 0:2]
                plt.subplot(2,2,1)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=1)
                plt.colorbar(label=r"$Y$")
                center = rrlda.means_.dot(W)
                plt.scatter(center[:, 0], center[:, 1], c=np.arange(self.c), cmap='RdBu_r', marker='x', s=500)
                for i in range(self.c):
                    plt.text(center[i, 0], center[i, 1], f"class {i}", fontsize=12, ha='center', va='center')

            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title("Reduced Rank Linear Discriminant Analysis (RRLDA)")

            # projection by multidimensional scaling (MDS)
            mds = sklearn.manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=None)
            X_trans = mds.fit_transform(self.X)
            plt.subplot(2,2,2)
            plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r')
            plt.xlabel("MDS Component 1")
            plt.ylabel("MDS Component 2")
            plt.title("Multidimensional Scaling (MDS)")
            plt.colorbar(label=r"$Y$")

            # projection by t-distributed stochastic neighbor embedding (t-SNE)
            tsne = sklearn.manifold.TSNE(n_components=2, random_state=None)
            X_trans = tsne.fit_transform(self.X)
            plt.subplot(2,2,3)
            plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r')
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.title("t-distributed Stochastic Neighbor Embedding (t-SNE)")
            plt.colorbar(label=r"$Y$")

            # projection by Uniform Manifold Approximation and Projection (UMAP)
            umap_ = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean")
            X_trans = umap_.fit_transform(self.X)
            plt.subplot(2,2,4)
            plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r')
            plt.xlabel("UMAP Component 1")
            plt.ylabel("UMAP Component 2")
            plt.title("Uniform Manifold Approximation and Projection (UMAP)")
            plt.colorbar(label=r"$Y$")
            plt.suptitle("Visualization of data by low-dimensional projection")
            plt.tight_layout()

    def normality(self):
        '''
        Diagnostic analysis on normality.
        Analysis include:
            1. Q-Q plot to check univariate normality feature by feature

            2. Box-Cox transformation for univariate normality
                Refer to linear regression for Box-Cox transformation in details.

            3. Yeo-Johnson transformation for univariate normality
                Refer to linear regression for Yeo-Johnson transformation in details.

            4. Mahalanobis distance to check multivariate normality
                D_i^2 = (X_i - mu)^T * \Sigma^(-1) * (X_i - mu)
                where X_i is the i-th observation, mu is the mean vector, \Sigma is the covariance matrix
                D_i^2 follows Chi-square distribution with p degrees of freedom

            5. Henze-Zirkler's test
                Henze-Zirkler's test is a multivariate normality test based on the empirical characteristic function
                HZ = n \int |\phi_n(t) - \phi(t)|^2 w(t) dt
                where n is the number of observations, \phi_n(t) is the empirical characteristic function, \phi(t) is the theoretical characteristic function, and w(t) is the weight function
                
            6. Mardia test
                Mardia test is a multivariate normality test based on skewness and kurtosis

        return:
            self.normality_test: dict
                - "box-cox": optimal lambda for Box-Cox transformation, shape (p,)
                - "yeo-johnson": optimal lambda for Yeo-Johnson transformation, shape (p,)
                - "henze-zirkler": list: [Henze-Zirkler's test p-value]
                - "mardia": list: [Mardia's skewness p-value, Mardia's kurtosis p-value]

        '''
        if not hasattr(self, "lda"):
            raise Exception("Fit the model first.")
        self.normality_test = {}

        # Q-Q plot to check univariate normality
        ncol = 4; nrow = self.p//ncol + 1
        plt.figure(figsize=(3*ncol, 3*nrow))
        for i in range(self.p):
            plt.subplot(nrow, ncol, i + 1)
            sm.qqplot(self.X[:, i], line="s", ax=plt.gca())
            plt.title(f"{self.X_columns[i]}")
        plt.suptitle(r"Q-Q plot of $X$")
        plt.tight_layout()

        # Box-Cox transformation for univariate normality
        box_cox_optimal = np.zeros(self.p); box_cox_optimal[:] = np.nan
        plt.figure(figsize=(6, 6))
        plt.subplot(2,1,1)
        for i in range(self.p):
            if any(self.X[:, i] <= 0):
                continue
            lambda_ = np.linspace(-10, 10, 1000)
            log_likelihood = np.zeros(lambda_.shape)
            for j in range(len(lambda_)):
                X_trans = scipy.stats.boxcox(self.X[:, i], lmbda=lambda_[j])
                residual = X_trans - np.mean(X_trans)
                log_likelihood[j] = -0.5*self.n*np.log(np.mean(np.power(residual, 2))) + (lambda_[j] - 1)*np.sum(np.log(self.X[:, i]))
            plt.plot(lambda_, log_likelihood, label=self.X_columns[i])
            #plt.vlines(lambda_[np.argmax(log_likelihood)], ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label=r"$\lambda^*$"+"=%.3f" % lambda_[np.argmax(log_likelihood)])
            box_cox_optimal[i] = lambda_[np.argmax(log_likelihood)]
        plt.xlabel("power"); plt.ylabel("log-likelihood")
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(self.X_columns, box_cox_optimal, "o-")
        plt.xticks(rotation=45)
        plt.axhline(1, color="black", linestyle="--")
        plt.ylabel(r"$\lambda^*$")
        plt.suptitle("Box-Cox transformation")
        plt.tight_layout()
        self.normality_test["box_cox"] = box_cox_optimal

        # Yeo-Johnson transformation for univariate normality
        yeo_johnson_optimal = np.zeros(self.p); yeo_johnson_optimal[:] = np.nan
        plt.figure(figsize=(6, 6))
        plt.subplot(2,1,1)
        for i in range(self.p):
            lambda_ = np.linspace(-10, 10, 1000)
            log_likelihood = np.zeros(lambda_.shape)
            for j in range(len(lambda_)):
                X_trans = scipy.stats.yeojohnson(self.X[:, i], lmbda=lambda_[j])
                pos_idx = np.where(self.X[:, i] >= 0)[0]; neg_idx = np.where(self.X[:, i] < 0)[0]
                residual = X_trans - np.mean(X_trans)
                log_likelihood[j] = -0.5*self.n*np.log(np.mean(np.power(residual, 2))) + (lambda_[j]-1)*np.sum(np.log(self.X[pos_idx, i]+1)) + (1-lambda_[j])*np.sum(np.log(-self.X[neg_idx, i]+1))
            plt.plot(lambda_, log_likelihood, label=self.X_columns[i])
            #plt.vlines(lambda_[np.argmax(log_likelihood)], ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label=r"$\lambda^*$"+"=%.3f" % lambda_[np.argmax(log_likelihood)])
            yeo_johnson_optimal[i] = lambda_[np.argmax(log_likelihood)]
        plt.xlabel("power"); plt.ylabel("log-likelihood")
        plt.legend(ncol=3)
        plt.subplot(2,1,2)
        plt.plot(self.X_columns, yeo_johnson_optimal, "o-")
        plt.xticks(rotation=45)
        plt.axhline(1, color="black", linestyle="--")
        plt.ylabel(r"$\lambda^*$")
        plt.suptitle("Yeo-Johnson transformation")
        plt.tight_layout()
        self.normality_test["yeo_johnson"] = yeo_johnson_optimal

        # Mahalanobis distance to check multivariate normality
        D_i = []
        Sigma = self.lda.covariance_
        for i in range(self.n):
            mu = self.lda.means_[self.Y[i, 0], :]
            diff = self.X[i, :] - mu
            D_i.append(diff.reshape(1, -1).dot(np.linalg.inv(Sigma)).dot(diff.reshape(-1, 1))[0][0])
        D_i = np.sort(np.array(D_i))
        chi_quantile = scipy.stats.chi2.ppf(np.linspace(0, 1, len(D_i)), self.p)
        plt.figure(figsize=(4, 4))
        plt.scatter(chi_quantile, D_i)
        plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), color="red", linestyle="--")
        plt.xlabel(r"Theoretical quantiles of $\chi^2_{p}$")
        plt.ylabel(r"Sample quantiles of $D_i$")
        plt.title("Q-Q plot of Mahalanobis distance")

        # Henze-Zirkler's test
        henzi_zirkler = pingouin.multivariate_normality(self.X, alpha=0.05)
        self.normality_test["henzi_zirkler"] = [henzi_zirkler.pval]
        print(f"Henze-Zirkler's test: HZ = {henzi_zirkler.hz}, p-value = {henzi_zirkler.pval}")

        # Mardia test
        def mardia_test(X):
            n, p = X.shape
            mean = np.mean(X, axis=0)
            S = np.cov(X, rowvar=False)
            S_inv = np.linalg.inv(S)
            X_centered = X - mean

            # Compute Mahalanobis distances
            D = np.array([x.dot(S_inv).dot(x.T) for x in X_centered])

            # Mardia's skewness
            skewness = np.sum([(x.dot(S_inv).dot(y.T))**3 for x in X_centered for y in X_centered]) / (n**2)
            skewness_stat = n * skewness / 6
            skewness_p = 1 - scipy.stats.chi2.cdf(skewness_stat, df=p*(p+1)*(p+2)/6)

            # Mardia's kurtosis
            kurtosis = np.sum(D**2) / n
            expected_kurtosis = p * (p + 2)
            kurtosis_stat = (kurtosis - expected_kurtosis) / np.sqrt(8 * p * (p + 2) / n)
            kurtosis_p = 2 * (1 - scipy.stats.norm.cdf(np.abs(kurtosis_stat)))
            return {
                'skewness': skewness,
                'skewness_stat': skewness_stat,
                'skewness_p_value': skewness_p,
                'kurtosis': kurtosis,
                'kurtosis_stat': kurtosis_stat,
                'kurtosis_p_value': kurtosis_p
            }
        mardia = mardia_test(self.X)
        self.normality_test["mardia"] = [mardia["skewness_p_value"], mardia["kurtosis_p_value"]]
        print(f"Mardia's skewness: {mardia['skewness']}, p-value = {mardia['skewness_p_value']}")
        print(f"Mardia's kurtosis: {mardia['kurtosis']}, p-value = {mardia['kurtosis_p_value']}")

    def equal_covariance(self):
        '''
        Diagnostic analysis on equal covariance.
        Analysis include:
            1. Box M test for equal covariance matrices
                Box M test is a multivariate test for equal covariance matrices across groups
                H_0: \Sigma_1 = \Sigma_2 = ... = \Sigma_k

            2. Univariate test for equal variances
                H_0: \sigma_i^2 = \sigma_j^2, i != j
            
            3. Likelihood ratio test via MANOVA
                H_0: \Sigma_1 = \Sigma_2 = ... = \Sigma_k

            4. Compare with Quadratic Discriminant Analysis (QDA)
                Compare the out-of-sample accuracy between LDA and QDA

        return:
            self.normality_test: dict
                - "box_m": list: [p-value for Box-M transformation]
                - "equal_variance_between_classes": p-value matrix for equal variance between classes, shape (c, c)
                - "manova": list: [p-value for MANOVA test]
                - "lda_qda_accuracy": out-of-sample accuracy of LDA and QDA, shape (2,)
        '''        
        if not hasattr(self, "lda"):
            raise Exception("Fit the model first.")
        self.equal_covariance = {}

        # Box M test for equal covariance matrices
        data = pd.DataFrame(self.X, columns=self.X_columns)
        data["target"] = self.Y.flatten()
        box_m = pingouin.box_m(data, dvs=self.X_columns, group="target")
        print(f"Box M test: Box M = {box_m.loc['box', 'Chi2']}, p-value = {box_m.loc['box', 'pval']}")
        self.equal_covariance["box_m"] = [box_m.loc["box", "pval"]]

        # Univariate test for equal variances
        equal_variance_between_classes = np.zeros((self.c, self.c)); equal_variance_between_classes[:] = np.nan 
        for i in range(self.c):
            for j in range(self.c):
                if i == j:
                    continue
                X1 = self.X[self.Y.flatten() == i, :]
                X2 = self.X[self.Y.flatten() == j, :]
                equal_variance_between_classes[i, j] = statsmodels.stats.multivariate.test_cov_oneway([np.cov(X1, rowvar=False), np.cov(X2, rowvar=False)], [X1.shape[0], X2.shape[0]]).pvalue
        self.equal_covariance["equal_variance_between_classes"] = equal_variance_between_classes
        plt.figure(figsize=(6, 6))
        colors = ['red', 'yellow', 'green']  # for <0.01, 0.01–0.05, >0.05
        bounds = [0, 0.01, 0.05, 1]
        cmap = matplotlib.colors.ListedColormap(colors)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(equal_variance_between_classes, cmap=cmap, norm=norm, aspect='auto')
        plt.colorbar(label="p-value")
        plt.xticks(range(self.c), [f"Class {i}" for i in range(self.c)], rotation=45)
        plt.yticks(range(self.c), [f"Class {i}" for i in range(self.c)])
        plt.title(r"p-value for $\Sigma_i = \Sigma_j$")

        # Likelihood ratio test via MANOVA
        data = pd.DataFrame(self.X, columns=self.X_columns)
        data["target"] = self.Y.flatten()
        manova = statsmodels.multivariate.manova.MANOVA.from_formula("+".join(self.X_columns) + " ~ target", data=data)
        result = manova.mv_test()
        for line in str(result).split("\n"):
            if "Wilks" in line:
                tokens = line.split()
                F_stats = float(tokens[2])
                p_value = float(tokens[6])
                break
        print(f"MANOVA test: F = {F_stats}, p-value = {p_value}")
        self.equal_covariance["manova"] = [p_value]

        # Compare with Quadratic Discriminant Analysis (QDA)
        lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="svd", store_covariance=False)
        lda.fit(self.X_train, self.Y_train.flatten())
        y_pred_lda = lda.predict(self.X_test)
        lda_accuracy = np.mean(y_pred_lda == self.Y_test.flatten())
        qda = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariance=False)
        qda.fit(self.X_train, self.Y_train.flatten())
        y_pred_qda = qda.predict(self.X_test)
        qda_accuracy = np.mean(y_pred_qda == self.Y_test.flatten())
        plt.figure(figsize=(6,3))
        plt.bar(["LDA", "QDA"], [lda_accuracy, qda_accuracy], color=['#1f77b4', '#ff7f0e'])
        plt.ylabel("Accuracy"); plt.title("Out-of-sample accuracy: LDA vs QDA")
        self.equal_covariance["lda_qda_accuracy"] = [lda_accuracy, qda_accuracy]

    def colinearity(self):
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
        corr = np.corrcoef(self.X, rowvar=False)
        self.colinearity_test["correlation"] = corr
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

        if not hasattr(self, "lda"):
            raise Exception("Fit the model first.")
        self.outlier_test = {}

        # Univariate standardization
        mu = np.zeros((self.c, self.p)); mu[:] = np.nan
        std = np.zeros((self.c, self.p)); std[:] = np.nan
        for j in range(self.p):
            for c in range(self.c):
                X = self.X[self.Y.flatten() == c, j]
                mu[c, j] = np.mean(X); std[c, j] = np.std(X)
        Z = np.zeros((self.n, self.p)); Z[:] = np.nan
        for i in range(self.n):
            for j in range(self.p):
                c = self.Y[i, 0]
                Z[i, j] = (self.X[i, j] - mu[c, j]) / std[c, j]
        Z = np.abs(Z)
        Z_ind = np.sum(Z > 3, axis=1)
        self.outlier_test["univariate_standardization"] = np.where(Z_ind > 0)[0]

        # Mahalanobis distance for multivariate normality
        mu = []; cov = []
        for c in range(self.c):
            X = self.X[self.Y.flatten() == c, :]
            mu.append(np.mean(X, axis=0))
            cov.append(np.cov(X, rowvar=False))
        D_i = np.zeros(self.n); D_i[:] = np.nan
        for i in range(self.n):
            mu_i = mu[self.Y[i, 0]]
            cov_i = cov[self.Y[i, 0]]
            diff = self.X[i, :] - mu_i
            D_i[i] = diff.reshape(1, -1).dot(np.linalg.inv(cov_i)).dot(diff.reshape(-1, 1))[0,0]
        threshold = scipy.stats.chi2.ppf(0.95, self.p)
        self.outlier_test["mahalanobis_dist"] = np.where(D_i > threshold)[0]
        if threshold == "strict":
            self.outlier_test["summary"] = np.union1d(self.outlier_test["univariate_standardization"], self.outlier_test["mahalanobis_dist"])
        else:
            self.outlier_test["summary"] = self.outlier_test["mahalanobis_dist"]

        if is_output:
            plt.figure(figsize=(6, 9))
            plt.subplot(3, 1, 1)
            plt.imshow(Z.T, cmap='Reds', vmin=0, vmax=3, aspect='auto')
            plt.colorbar(label=r"$|\frac{X_{ij}-\mu_{k,j}}{\sigma_{k,j}}|$")
            plt.xlabel("observation index"); plt.ylabel("feature index")
            plt.title("Univariate standardization")
            plt.subplot(3, 1, 2)
            plt.scatter(range(self.n), Z_ind, s=1, label=r"$\sum_{j=1}^p |\frac{X_{ij}-\mu_{k,j}}{\sigma_{k,j}}| > 3$")
            plt.legend()
            plt.xlabel("observation index")
            plt.title("Univariate standardization")
            plt.subplot(3, 1, 3)
            plt.scatter(range(self.n), D_i, s=1, label="Mahalanobis distance")
            plt.scatter(self.outlier_test["mahalanobis_dist"], D_i[self.outlier_test["mahalanobis_dist"]], s=1, color="red", label="outliers")
            plt.axhline(threshold, color="red", linestyle="--", label="threshold", alpha=0.5)
            plt.xlabel("Observation index")
            plt.ylabel("Mahalanobis distance")
            plt.title("Mahalanobis Distance")
            plt.legend(loc="lower right")
            plt.tight_layout()
        
        return self.outlier_test

    def feature_selection(self, criterion="accuracy (out-of-sample)"):
        '''
        Feature selection using different methods:
            (1) Best subset selection (self.feature_selection_best_subset)
            (2) Forward stepwise selection (self.feature_selection_forward_stepwise)

        return:
            self.feature_selection_best_subset_summary: dict
                - "feature_number": [feature_idx, accuracy (in-sample), AIC (in-sample), BIC (in-sample), accuracy (out-of-sample)]
            self.feature_selection_forward_stepwise_summary: dict
                - "feature_number": [feature_idx, accuracy (in-sample), AIC (in-sample), BIC (in-sample), accuracy (out-of-sample)]
        '''
        if not hasattr(self, "lda"):
            raise Exception("Fit the model first.")
        self.selected_feature = collections.defaultdict(list)
        
        # Univarite class seperation
        univariate_anova = np.zeros(self.p); univariate_anova[:] = np.nan
        for i in range(self.p):
            f, p = scipy.stats.f_oneway(*[self.X[self.Y.flatten() == c, i] for c in range(self.c)])
            univariate_anova[i] = np.max([p, 1e-4])
        plt.figure(figsize=(6, 4))
        plt.bar(range(self.p), univariate_anova, color=["red" if p < 0.01 else "orange" if p < 0.05 else "green" for p in univariate_anova])
        plt.xticks(range(self.p), self.X_columns, rotation=45)
        plt.ylabel("max(p-value, 1e-4)")
        plt.ylim(1e-5, 1); plt.yscale("log")
        plt.fill_between(range(self.p), 1e-5, 0.01, color="red", alpha=0.2, label="p < 0.01, effective seperation")
        plt.fill_between(range(self.p), 0.01, 0.05, color="orange", alpha=0.2, label="0.01 < p < 0.05, medium seperation")
        plt.fill_between(range(self.p), 0.05, 1, color="green", alpha=0.2, label="p > 0.05, weak seperation")
        plt.title("p-value for univariate multi-class ANOVA")
        plt.legend()

        # best subset selection
        print("--- Feature selection by best subset ---")
        self.feature_selection_best_subset_summary = collections.defaultdict(list)
        self.feature_selection_best_subset_summary["feature_number"] = ["feature_idx", "accuracy (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "accuracy (out-of-sample)"]
        
        log = collections.defaultdict(list)
        for feature_num in range(1, self.p + 1):
            for feature_idx in itertools.combinations(range(0, self.p), feature_num):
                feature_idx = list(feature_idx)
                X_trans = self.X[:, feature_idx].reshape(-1, len(feature_idx))
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
                lda.fit(X_trans, self.Y.flatten())
                accuracy_in_sample = np.mean(lda.predict(X_trans) == self.Y.flatten())
                log_likelihood = 0
                for c in range(self.c):
                    X_c = X_trans[self.Y.flatten() == c, :]
                    n_c = X_c.shape[0]
                    mu_c = lda.means_[c, :]
                    log_likelihood += np.sum(np.log(n_c / self.n) + scipy.stats.multivariate_normal.logpdf(X_c, mean=mu_c, cov=lda.covariance_))
                k = self.c*X_trans.shape[1] + X_trans.shape[1]*(X_trans.shape[1]+1)/2 + self.c - 1
                aic = 2*k - 2*log_likelihood
                bic = np.log(self.n)*k - 2*log_likelihood

                X_train_trans = self.X_train[:, feature_idx].reshape(-1, len(feature_idx))
                X_test_trans = self.X_test[:, feature_idx].reshape(-1, len(feature_idx))
                lda_oos = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="svd", store_covariance=False)
                lda_oos.fit(X_train_trans, self.Y_train.flatten())
                accuracy_out_of_sample = np.mean(lda_oos.predict(X_test_trans) == self.Y_test.flatten())
                log[feature_num].append([feature_idx, accuracy_in_sample, aic, bic, accuracy_out_of_sample])
        
        plt.figure(figsize=(8, 6))
        plt.subplot(4, 1, 1)
        for feature_num in np.arange(1, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[1])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[1] for i in log[feature_num]], "o", color="gray")
            if criterion == "accuracy (in-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(1, self.p + 1), [log[i][0][1] for i in np.arange(1, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("accuracy \n(in-sample)")

        plt.subplot(4, 1, 2)
        for feature_num in np.arange(1, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[2])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[2] for i in log[feature_num]], "o", color="gray")
            if criterion == "AIC (in-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(1, self.p + 1), [log[i][0][2] for i in np.arange(1, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("AIC \n(in-sample)")

        plt.subplot(4, 1, 3)
        for feature_num in np.arange(1, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[3])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[3] for i in log[feature_num]], "o", color="gray")
            if criterion == "BIC (in-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(1, self.p + 1), [log[i][0][3] for i in np.arange(1, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("BIC \n(in-sample)")

        plt.subplot(4, 1, 4)
        for feature_num in np.arange(1, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[4])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[4] for i in log[feature_num]], "o", color="gray")
            if criterion == "accuracy (out-of-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(1, self.p + 1), [log[i][0][4] for i in np.arange(1, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("$R^2$ \n (out-of-sample)")
        plt.suptitle("Feature selection by best subset selection with criterion: %s" % criterion)
        plt.tight_layout()

        # forward stepwise selection
        print("--- Feature selection by forward stepwise ---")
        self.feature_selection_forward_stepwise_summary = collections.defaultdict(list)
        selected_feature = set()
        log = collections.defaultdict(list)
        for feature_num in np.arange(1, self.p+1, 1):
            for feature_idx in range(self.p):
                if feature_idx not in selected_feature:
                    X_trans = self.X[:, list(selected_feature) + [feature_idx]].reshape(-1, len(selected_feature) + 1)
                    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
                    lda.fit(X_trans, self.Y.flatten())
                    accuracy_in_sample = np.mean(lda.predict(X_trans) == self.Y.flatten())
                    log_likelihood = 0
                    for c in range(self.c):
                        X_c = X_trans[self.Y.flatten() == c, :]
                        n_c = X_c.shape[0]
                        mu_c = lda.means_[c, :]
                        log_likelihood += np.sum(np.log(n_c / self.n) + scipy.stats.multivariate_normal.logpdf(X_c, mean=mu_c, cov=lda.covariance_))
                    k = self.c*X_trans.shape[1] + X_trans.shape[1]*(X_trans.shape[1]+1)/2 + self.c - 1
                    aic = 2*k - 2*log_likelihood
                    bic = np.log(self.n)*k - 2*log_likelihood

                    X_train_trans = self.X_train[:, list(selected_feature) + [feature_idx]].reshape(-1, len(selected_feature) + 1)
                    X_test_trans = self.X_test[:, list(selected_feature) + [feature_idx]].reshape(-1, len(selected_feature) + 1)
                    lda_oos = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="svd", store_covariance=False)
                    lda_oos.fit(X_train_trans, self.Y_train.flatten())
                    accuracy_out_of_sample = np.mean(lda_oos.predict(X_test_trans) == self.Y_test.flatten())
                    log[feature_num].append([list(selected_feature) + [feature_idx], accuracy_in_sample, aic, bic, accuracy_out_of_sample])
            if criterion == "accuracy (in-sample)":
                log[feature_num].sort(reverse=True, key=lambda x: x[1])
            if criterion == "AIC (in-sample)":
                log[feature_num].sort(reverse=False, key=lambda x: x[2])
            if criterion == "BIC (in-sample)":
                log[feature_num].sort(reverse=False, key=lambda x: x[3])
            if criterion == "accuracy (out-of-sample)":
                log[feature_num].sort(reverse=True, key=lambda x: x[4])
            selected_feature.add(log[feature_num][0][0][-1])
            print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
            self.feature_selection_forward_stepwise_summary[feature_num] = log[feature_num][0]

        plt.figure(figsize=(8, 6))
        plt.subplot(4,1,1)
        plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][1] for i in np.arange(1, self.p+1, 1)], "o-")
        plt.xlabel("feature number"); plt.ylabel("accuracy \n (in-sample)")
        plt.subplot(4,1,2)
        plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][2] for i in np.arange(1, self.p+1, 1)], "o-")
        plt.xlabel("feature number"); plt.ylabel("AIC \n(in-sample)")
        plt.subplot(4,1,3)
        plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][3] for i in np.arange(1, self.p+1, 1)], "o-")
        plt.xlabel("feature number"); plt.ylabel("BIC \n(in-sample)")
        plt.subplot(4,1,4)
        plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][4] for i in np.arange(1, self.p+1, 1)], "o-")
        plt.xlabel("feature number"); plt.ylabel("accuracy \n(out-of-sample)")
        plt.suptitle("Feature selection by forward stepwise selection with criterion: %s" % criterion)
        plt.tight_layout()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        ar = np.zeros((self.p, self.p)); ar[:] = np.nan
        for feature_num in np.arange(1, self.p+1, 1):
            ar[feature_num-1, self.feature_selection_best_subset_summary[feature_num][0]] = 1
        plt.imshow(ar, cmap='Blues', vmin=0, vmax=1, alpha=0.5, aspect='auto')
        for i in range(self.p):
            plt.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5)
            plt.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(range(self.p), self.X_columns, rotation=45)
        plt.yticks(range(self.p), range(1, self.p+1))
        plt.xlabel("selected feature")
        plt.ylabel("feature number")
        plt.title("Best subset selection")

        plt.subplot(1, 2, 2)
        ar = np.zeros((self.p, self.p)); ar[:] = np.nan
        for feature_num in np.arange(1, self.p+1, 1):
            ar[feature_num-1, self.feature_selection_forward_stepwise_summary[feature_num][0]] = 1
        plt.imshow(ar, cmap='Blues', vmin=0, vmax=1, alpha=0.5, aspect='auto')
        for i in range(self.p):
            plt.axvline(x=i-0.5, color='gray', linestyle='--', linewidth=0.5)
            plt.axhline(y=i-0.5, color='gray', linestyle='--', linewidth=0.5)
        plt.xticks(range(self.p), self.X_columns, rotation=45)
        plt.yticks(range(self.p), range(1, self.p+1))
        plt.xlabel("selected feature")
        plt.ylabel("feature number")
        plt.title("Forward stepwise selection")
        plt.suptitle("Feature selection by best subset and forward stepwise selection")
        plt.tight_layout()

    def optimal_shrinkage_covariance(self, is_output=True):
        '''
        Search for optimal shrinkage parameter for covariance matrix estimation by cross-validation.
        Compare with Ledoit-Wolf method.
        Reference:
            Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of the Royal Statistical Society: Series A (Statistics in Society), 68(1), 187-220.

        return:
            optimal_shrinkage_covariance: optimal shrinkage parameter for covariance matrix estimation
                - "auto": Ledoit-Wolf method
                - float: optimal shrinkage parameter by cross-validation
        '''
        self.optimal_shrinkage_covariance = {}
        lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="eigen", store_covariance=True, shrinkage='auto')
        lda.fit(self.X_train, self.Y_train.flatten())
        y_pred = lda.predict(self.X_test)
        benchmark = np.mean(y_pred == self.Y_test.flatten())

        alpha_ = np.linspace(0, 1, 100); accuracy = np.zeros(len(alpha_)); accuracy[:] = np.nan
        for i in range(len(alpha_)):
            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="eigen", store_covariance=False, shrinkage=alpha_[i])
            lda.fit(self.X_train, self.Y_train.flatten())
            y_pred = lda.predict(self.X_test)
            accuracy[i] = np.mean(y_pred == self.Y_test.flatten())
        if is_output:
            plt.figure(figsize=(6, 3))
            plt.plot(alpha_, accuracy)
            plt.xlabel("shrinkage parameter"); plt.ylabel("accuracy (out-of-sample)")
            plt.axhline(y=benchmark, color='red', linestyle='--', label="benchmark (Ledoit-Wolf)")
            plt.legend()
        
        return alpha_[np.argmax(accuracy)] if np.max(accuracy) > benchmark else 'auto'

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/vowel_train.csv"), index_col=0)
X = data.iloc[:, 1:].to_numpy()
Y = data.iloc[:, 0].to_numpy().reshape(-1, 1) - 1
idx = np.where(Y.flatten() <= 1)[0]
X = X[idx, :]; Y = Y[idx, :]
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/vowel_test.csv"), index_col=0)
X_test = data.iloc[:, 1:].to_numpy()
Y_test = data.iloc[:, 0].to_numpy().reshape(-1, 1) - 1
idx = np.where(Y_test.flatten() <= 1)[0]
X_test = X_test[idx, :]; Y_test = Y_test[idx, :]

#model = linear_discriminant_analysis(X, Y, X_test=X_test, Y_test=Y_test, is_normalize=True, test_size_ratio=0.2)
#model.fit(solver="eigen", is_output=True)
#model.visualize_data()
#model.normality()
#model.equal_covariance()
#model.colinearity()
#model.outlier(threshold="strict", is_output=True)
#model.feature_selection()
#model.optimal_shrinkage_covariance(is_output=True)

#%%
class logistic_regression:
    def __init__(self, X, Y, X_test=None, Y_test=None, X_columns=None, is_normalize=True, test_size_ratio=0.2):
        '''
        Initialize the logistic regression model.
        params:
            X: feature matrix, shape (n, p)
            Y: target vector, shape (n, 1). Y is a vector encoding class information, with the first class as 0, the second class as 1, and so on.
            X_test: test feature matrix, shape (n_test, p)
            Y_test: test target vector, shape (n_test, 1)
                If X_test and Y_test are not provided, the data will be split into train and test sets based on the test_size_ratio.
            X_columns: feature names, list of length p. If None, use default names ["feature_0", "feature_1", ...]
            is_normalize: whether to normalize the data
            test_size_ratio: the ratio of test set size to the total size
            binary_class: whether the model is for binary classification. If True, the model will be fitted as a binary logistic regression model.

        General workflow:
            (1) Fit logistic regression model;
            (2) Check for nonlinearity in logit;
            (3) Check for multicollinearity of features;
            (4) Check for separation or quasi-separation;
            (5) Check for influential observations;
            (6) Feature selection;
            (7) Model validation, refinement, and optimization
        '''
        self.X = X; self.Y = Y.reshape(-1, 1)
        self.X_columns = X_columns if X_columns is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.is_normalize = is_normalize
        self.test_size_ratio = test_size_ratio

        self.n = X.shape[0]; self.p = X.shape[1]; self.c = len(np.unique(Y.flatten()))

        if self.is_normalize:
            self.X_mean = np.mean(self.X, axis=0)
            self.X_std = np.std(self.X, axis=0, ddof=1)
            self.X = (self.X - self.X_mean) / self.X_std

        if (X_test is not None) and (Y_test is not None):
            self.X_train = self.X
            self.Y_train = self.Y
            self.X_test = X_test.copy()
            self.Y_test = Y_test.copy().reshape(-1, 1)
            if self.is_normalize:
                self.X_test = (X_test.copy() - self.X_mean) / self.X_std
        else:
            self.X_train = self.X.copy()[0:int(self.n*(1-self.test_size_ratio)), :]
            self.Y_train = self.Y.copy()[0:int(self.n*(1-self.test_size_ratio)), :]
            self.X_test = self.X.copy()[int(self.n*(1-self.test_size_ratio)):, :]
            self.Y_test = self.Y.copy()[int(self.n*(1-self.test_size_ratio)):, :]

    def fit(self, is_output=True):
        '''
        Fit the logistic regression model.
        params:
            is_output: whether to print the summary of the model

        return:
            self.logit: sklearn logistic regression model
        '''
        if self.c == 2:
            self.logit = statsmodels.api.Logit(self.Y.flatten(), self.X).fit()
            if is_output:
                print(self.logit.summary())
                plt.figure(figsize=(6, 6))
                plt.subplot(2, 1, 1)
                plt.errorbar(self.X_columns, self.logit.params, yerr=self.logit.bse, fmt="o-")
                plt.axhline(0, color="black", linestyle="--")
                plt.xticks(rotation=45); plt.ylabel(r"$\beta$")
                plt.subplot(2, 1, 2)
                plt.plot(self.X_columns, self.logit.pvalues, "o-")
                plt.axhline(0.05, color="red", linestyle="--", label="0.05")
                plt.axhline(0.01, color="orange", linestyle="--", label="0.01")
                plt.ylim(1e-3, 1); plt.yscale("log")
                plt.legend()
                plt.xticks(rotation=45)
                plt.ylabel("p-value")
                plt.suptitle("Logistic regression coefficients")
                plt.tight_layout()
        else:
            self.logit = statsmodels.api.MNLogit(self.Y.flatten(), self.X).fit()
            if is_output:
                print(self.logit.summary())
                ncol = 3; nrow = int(np.ceil(self.c / ncol))
                plt.figure(figsize=(4*ncol, 2*nrow))
                for c in range(self.c-1):
                    plt.subplot(ncol, nrow, c+1)
                    plt.errorbar(self.X_columns, self.logit.params[c, :], yerr=self.logit.bse[c, :], fmt="o-", label=f"Class {c}")
                    plt.axhline(0, color="black", linestyle="--")
                    plt.xticks(rotation=45); plt.ylabel(r"$\beta$")
                    plt.legend()
            plt.tight_layout()

            p_value = np.zeros((self.c-1, self.p)); p_value[:] = np.nan
            for c in range(self.c-1):
                p_value[c, :] = self.logit.pvalues[c, :]
            plt.figure(figsize=(6, 6))
            colors = ['red', 'yellow', 'green']
            bounds = [0, 0.01, 0.05, 1]
            cmap = matplotlib.colors.ListedColormap(colors)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            plt.imshow(p_value, cmap=cmap, norm=norm, aspect='auto')
            plt.xticks(range(self.p), self.X_columns, rotation=45)
            plt.yticks(range(self.c-1), [f"Class {i}" for i in range(self.c-1)])
            plt.colorbar(ticks=[0, 0.01, 0.05, 1], label="p-value")

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/vowel_train.csv"), index_col=0)
X = data.iloc[:, 1:].to_numpy()
Y = data.iloc[:, 0].to_numpy().reshape(-1, 1) - 1
idx = np.where(Y.flatten() <= 1)[0]
X = X[idx, :]; Y = Y[idx, :]
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/vowel_test.csv"), index_col=0)
X_test = data.iloc[:, 1:].to_numpy()
Y_test = data.iloc[:, 0].to_numpy().reshape(-1, 1) - 1
idx = np.where(Y_test.flatten() <= 1)[0]
X_test = X_test[idx, :]; Y_test = Y_test[idx, :]

model = logistic_regression(X, Y, X_test=X_test, Y_test=Y_test, is_normalize=True, test_size_ratio=0.2)
model.fit(is_output=True)




# %%
