#%%
import os, sys, copy, scipy, datetime, tqdm, collections, itertools, warnings, contextlib
from turtle import color
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
import statsmodels.tools

import sklearn.linear_model
import sklearn.cross_decomposition
import sklearn.manifold
import sklearn.discriminant_analysis
import sklearn.utils.multiclass

import umap, pingouin

@contextlib.contextmanager
def suppress_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

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
            if self.is_normalize:
                self.X_train_mean = np.mean(self.X_train, axis=0)
                self.X_train_std = np.std(self.X_train, axis=0, ddof=1)
                self.X_train = (self.X_train - self.X_train_mean) / self.X_train_std
                self.X_test = (self.X_test - self.X_train_mean) / self.X_train_std

    def fit(self, solver="eigen"):
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

    def visualize_data(self, methods=["pandas_describe", "raw_data_plot", "boxplot", "reduce_rank_LDA", "MDS", "tSNE", "UMAP"]):
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
                    plt.plot(self.X[:, i] * self.X_std[i] + self.X_mean[i], color="blue")
                else:
                    plt.plot(self.X[:, i], color="blue")
                plt.ylabel(self.X_columns[i])
            plt.subplot(nrow, ncol, self.p + 1)
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
            plt.boxplot(self.Y, widths=0.5, showfliers=True)
            plt.ylabel("target")
            plt.suptitle("Boxplot of (unnormalized) features")
            plt.tight_layout()

        if self.p > 2 and any(["reduce_rank_LDA" in methods, "MDS" in methods, "tSNE" in methods, "UMAP" in methods]):
            color_feature = self.Y.flatten()
            plt.figure(figsize=(12, 12))

            # projection by reduced rank linear discriminant analysis (RRLDA)
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
            plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=1)
            plt.xlabel("MDS Component 1")
            plt.ylabel("MDS Component 2")
            plt.title("Multidimensional Scaling (MDS)")
            plt.colorbar(label=r"$Y$")

            # projection by t-distributed stochastic neighbor embedding (t-SNE)
            tsne = sklearn.manifold.TSNE(n_components=2, random_state=None)
            X_trans = tsne.fit_transform(self.X)
            plt.subplot(2,2,3)
            plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=1)
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.title("t-distributed Stochastic Neighbor Embedding (t-SNE)")
            plt.colorbar(label=r"$Y$")

            # projection by Uniform Manifold Approximation and Projection (UMAP)
            umap_ = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean")
            X_trans = umap_.fit_transform(self.X)
            plt.subplot(2,2,4)
            plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=1)
            plt.xlabel("UMAP Component 1")
            plt.ylabel("UMAP Component 2")
            plt.title("Uniform Manifold Approximation and Projection (UMAP)")
            plt.colorbar(label=r"$Y$")
            plt.suptitle("Visualization of data by low-dimensional projection")
            plt.tight_layout()

    def normality(self, method=["Q-Q_plot", "Box-Cox", "Yeo-Johnson", "Mahalanobis_distance", "test"]):
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

        if "Q-Q_plot" in method:
            # Q-Q plot to check univariate normality
            ncol = 4; nrow = self.p//ncol + 1
            plt.figure(figsize=(3*ncol, 3*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                sm.qqplot(self.X[:, i], line="s", ax=plt.gca())
                plt.title(f"{self.X_columns[i]}")
            plt.suptitle(r"Q-Q plot of $X$")
            plt.tight_layout()

        if "Box-Cox" in method and any([self.X[:, i].min() > 0 for i in range(self.p)]):
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

        if "Yeo-Johnson" in method:
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
        if "Mahalanobis_distance" in method:
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
        if "test" in method:
            henzi_zirkler = pingouin.multivariate_normality(self.X, alpha=0.05)
            self.normality_test["henzi_zirkler"] = [henzi_zirkler.pval]
            #print(f"Henze-Zirkler's test: HZ = {henzi_zirkler.hz}, p-value = {henzi_zirkler.pval}")

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
            #print(f"Mardia's skewness: {mardia['skewness']}, p-value = {mardia['skewness_p_value']}")
            #print(f"Mardia's kurtosis: {mardia['kurtosis']}, p-value = {mardia['kurtosis_p_value']}")
            plt.figure(figsize=(4, 2))
            labels = ["Henzi-Zirkler", "Mardia's skewness", "Mardia's kurtosis"]
            values = [henzi_zirkler.pval, mardia["skewness_p_value"], mardia["kurtosis_p_value"]]
            plt.barh(labels, [max(v,1e-3) for v in values], color=["green" if pval > 0.05 else "orange" if pval > 0.01 else "red" for pval in values])
            plt.axvline(0.05, color="green", linestyle="--")
            plt.axvline(0.01, color="orange", linestyle="--", label="normality")
            plt.xlabel("p-value")
            plt.xlim(1e-4, 1); plt.xscale("log")
            plt.legend()
            plt.title("Normality Test Results")

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
        corr = np.corrcoef(self.X, rowvar=False)
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

        if "eigenvalue" in method:
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

    def outlier(self, is_output=True):
        '''
        Diagnostic analysis on outliers.
        Analysis include:
            (1) Univariate standardization
                Z_ij = |(X_ij - \mu_{k,j})/\sigma_{k,j}|
                where X_ij is the i-th observation of j-th feature, \mu_{k,j} is the mean of j-th feature in class k, \sigma_{k,j} is the standard deviation of j-th feature in class k.
                Z_ij > 3 is considered as an outlier.
            (2) Mahalanobis distance for multivariate normality
                D_i^2 = (X_i - \mu_k)^T * \Sigma_k^(-1) * (X_i - \mu_k)
                where X_i is the i-th observation, \mu_k is the mean vector of class k, \Sigma_k is the covariance matrix of class k.
                D_i^2 follows Chi-square distribution with p degrees of freedom.

        params:
            threshold: str, defines the threshold for outlier detection.
                - "strict": 2 standard deviations
                - "loose": 3 standard deviations
            is_output: whether to print the summary of the outliers

        return:
            self.outlier_test: dict
                - "univariate_standardization": observation index for outliers by univariate standardization
                - "mahalanobis_dist": observation index for outliers by Mahalanobis distance
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
            plt.scatter(range(self.n), Z_ind, s=1, label=r"$\sum_{j=1}^p |\frac{X_{ij}-\mu_{k,j}}{\sigma_{k,j}}| > 3$",
                         color=["red" if i in self.outlier_test["univariate_standardization"] else "blue" for i in range(self.n)])
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
#model.fit(solver="eigen")
#model.visualize_data()
#model.normality()
#model.equal_covariance()
#model.colinearity()
#model.outlier(is_output=True)
#model.feature_selection()
#model.optimal_shrinkage_covariance(is_output=True)

#%%
class logistic_regression_binary:
    def __init__(self, X, Y, X_test=None, Y_test=None, X_columns=None, is_normalize=True, test_size_ratio=0.2):
        '''
        Initialize the logistic regression model for binary classification.
        params:
            X: feature matrix, shape (n, p)
            Y: target vector, shape (n, 1). Y is a vector encoding class information, with the first class as 0, the second class as 1.
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
        if sorted(np.unique(Y.flatten()).tolist()) != [0, 1]:
            raise ValueError("Y must be a binary vector with values 0 and 1.")
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
            if self.is_normalize:
                self.X_train_mean = np.mean(self.X_train, axis=0)
                self.X_train_std = np.std(self.X_train, axis=0, ddof=1)
                self.X_train = (self.X_train - self.X_train_mean) / self.X_train_std
                self.X_test = (self.X_test - self.X_train_mean) / self.X_train_std

    def fit(self, is_output=True):
        '''
        Fit the logistic regression model.
        params:
            is_output: whether to print and plot the summary of the model
        '''
        self.logit = self._fit_logit(sm.add_constant(self.X, prepend=True), self.Y, disp=is_output)
        if is_output and hasattr(self.logit, "summary"):
            print(self.logit.summary())
            plt.figure(figsize=(6, 6))
            plt.subplot(2, 1, 1)
            plt.errorbar(self.X_columns, self.logit.params[1:], yerr=self.logit.bse[1:], fmt="o-")
            plt.axhline(0, color="black", linestyle="--")
            plt.xticks(rotation=45); plt.ylabel(r"$\beta$")
            plt.subplot(2, 1, 2)
            plt.bar(self.X_columns, self.logit.pvalues[1:], color=["red" if p < 0.01 else "orange" if p < 0.05 else "green" for p in self.logit.pvalues[1:]])
            plt.axhline(0.01, color="red", linestyle="--", label="0.01")
            plt.axhline(0.05, color="orange", linestyle="--", label="0.05")
            plt.ylim(1e-4, 1); plt.yscale("log")
            plt.xticks(rotation=45)
            plt.ylabel("p-value")
            plt.suptitle("Logistic regression coefficients")
            plt.tight_layout()
        if is_output and hasattr(self.logit, "coef_"):
            plt.figure(figsize=(6, 3))
            plt.plot(self.X_columns, self.logit.coef_, fmt="o-")                
    
    def fit_regularized(self, penalty="l2", alpha=None, is_output=True):
        '''
        Fit the logistic regression model with regularization.
        params:
            penalty: type of regularization, "l1" for Lasso, "l2" for Ridge
            alpha: regularization strength. If None, search for optimal alpha by cross-validation.
            is_output: whether to print and plot the summary of the model
        '''
        if alpha == None:
            alpha_list = np.logspace(-6, 6, num=10000, base=10)
            accuracy = []
            for alpha in alpha_list:
                if type == "l1":
                    logit = sklearn.linear_model.LogisticRegression(penalty=penalty, C=1/alpha, solver="liblinear", max_iter=10000, fit_intercept=True)
                if type == "l2":
                    logit = sklearn.linear_model.LogisticRegression(penalty=penalty, C=1/alpha, solver="saga", max_iter=10000, fit_intercept=True)
                logit.fit(self.X_train, self.Y_train.flatten())
                Y_pred = logit.predict(self.X_test).reshape(-1, 1)
                Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                accuracy.append(1 - np.mean(np.abs(Y_pred_binary - self.Y_test)))
            alpha = alpha_list[np.argmax(accuracy)]

        if type == "l1":
            self.logit = sklearn.linear_model.LogisticRegression(penalty=penalty, C=1/alpha, solver="liblinear", max_iter=10000, fit_intercept=True)
        if type == "l2":
            self.logit = sklearn.linear_model.LogisticRegression(penalty=penalty, C=1/alpha, solver="saga", max_iter=10000, fit_intercept=True)
        self.logit.fit(self.X_train, self.Y_train.flatten())
        if is_output:
            plt.figure(figsize=(6, 3))
            plt.plot(self.X_columns, self.logit.coef_.flatten(), fmt="o-")
            plt.legend(title="alpha: %.4f" % (alpha))

    def predict(self, X):
        '''
        Predict the probability for class 1 and predicted class for the given input features.
        params:
            X: feature matrix, shape (n, p)
                Input data with no intercept term. The function will automatically add a constant term for the intercept.
        return:
            Y_pred: predicted probabilities for class 1, shape (n, 1)
            Y_pred_binary: predicted class labels, shape (n, 1)
                Binary class labels (0 or 1) based on the predicted probabilities.
        '''
        if not hasattr(self, "logit"):
            raise Exception("Fit the model first.")
        if self.is_normalize:
            X = (X - self.X_mean) / self.X_std
        Y_pred = self.logit.predict(sm.add_constant(X, prepend=True)).reshape(-1, 1)
        Y_pred_binary = (Y_pred.flatten() > 0.5).astype(int).reshape(-1, 1)
        return (Y_pred, Y_pred_binary)

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
            plt.boxplot(self.Y, widths=0.5, showfliers=True)
            plt.ylabel("target")
            plt.suptitle("Boxplot of (unnormalized) features")
            plt.tight_layout()

        if self.p > 2 and any(["PCA" in methods, "MDS" in methods, "tSNE" in methods, "UMAP" in methods]):
            color_feature = self.Y.flatten()
            plt.figure(figsize=(12, 12))
            plot_alpha = 0.8
            if "PCA" in methods:
                # projection by principal component analysis (PCA)
                U, S, VT = np.linalg.svd(self.X, full_matrices=False)
                V = VT.T; V = V[:, 0:2]
                X_trans = self.X.dot(V)
                plt.subplot(2,2,1)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=plot_alpha)
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.title("Principal component analysis (PCA)")
                plt.colorbar(label=r"$Y$")

            if "MDS" in methods:
                # projection by multidimensional scaling (MDS)
                mds = sklearn.manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=None)
                X_trans = mds.fit_transform(self.X)
                plt.subplot(2,2,2)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=plot_alpha)
                plt.xlabel("MDS Component 1")
                plt.ylabel("MDS Component 2")
                plt.title("Multidimensional Scaling (MDS)")
                plt.colorbar(label=r"$Y$")

            if "tSNE" in methods:
                # projection by t-distributed stochastic neighbor embedding (t-SNE)
                tsne = sklearn.manifold.TSNE(n_components=2, random_state=None)
                X_trans = tsne.fit_transform(self.X)
                plt.subplot(2,2,3)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=plot_alpha)
                plt.xlabel("t-SNE Component 1")
                plt.ylabel("t-SNE Component 2")
                plt.title("t-distributed Stochastic Neighbor Embedding (t-SNE)")
                plt.colorbar(label=r"$Y$")
            
            if "UMAP" in methods:
                # projection by Uniform Manifold Approximation and Projection (UMAP)
                umap_ = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean")
                X_trans = umap_.fit_transform(self.X)
                plt.subplot(2,2,4)
                plt.scatter(X_trans[:, 0], X_trans[:, 1], c=color_feature, cmap='RdBu_r', alpha=plot_alpha)
                plt.xlabel("UMAP Component 1")
                plt.ylabel("UMAP Component 2")
                plt.title("Uniform Manifold Approximation and Projection (UMAP)")
                plt.colorbar(label=r"$Y$")
            plt.suptitle("Visualization of data by low-dimensional projection")
            plt.tight_layout()

    def nonlinearity(self, smoother_type="polynomial", method=["binned_residual", "residual", "partial_residual", "interaction_term"]):
        '''
        Diagonostic analysis on non-linearity for logit(p_1/(1-p_1)).

        Analysis include:
            (1) binned residual plot
                Sort the predicted probabilities and group them into bins. 
                Plot the average predicted probabilities vs the average observed values in each bin.
                For binary classification, the plot should be close to a 45-degree line.

            (2) residual versus features plot
                Plot the estimated logit(p_1/(1-p_1)) vs each feature. Curvature in the plot suggests nonlinearity.

            (3) Partial residual plot
                Partial residual = deviance residual + estimated coefficient * feature, where
                    For binary classification, the deviance residual is defined as
                        r_i^{(\text{dev})} = \test{sign}(y_i - \hat{p}_i) \cdot \sqrt{2 \left[ y_i \log\left( \frac{y_i}{\hat{p}_i} \right) + (1 - y_i) \log\left( \frac{1 - y_i}{1 - \hat{p}_i} \right) \right]},
                    For multi-class classification, the deviance residual is defined as
                        r_i^{(\text{dev})} = \sqrt{-2 \log \hat{p}_{i,y_i}}.
                If the plot shows curvature, it suggests non-linearity.

            (4) Compare with models that include interaction term
                Add term x_i*x_j in model and compare the model metric
                    p-value of likelihood ratio test:
                        H_0: small model. 
                        If p is small, we reject the small model and proceed with the larger model. 
                    AIC, BIC, accuracy (out-of-sample): accept large model when we observe a significant reduction in AIC or BIC, or a significant increase in accuracy_oos.

        params:
            smoother_type: type for smoother in the plots. See self._smoother for details.

        returns:
            self.nonlinearity_test: dict
                - "interaction_term_metric" -- [p-value of F-statistics, AIC, BIC, accuracy (out-of-sample)]
        '''
        if not hasattr(self, "logit"):
            raise Exception("Fit the model first.")
        self.nonlinearity_test = {}

        # binned residual plot
        if "binned_residual" in method:
            bins = 20; batch_size = int(np.ceil(self.n / bins))
            Y_pred = self.logit.predict(sm.add_constant(self.X, prepend=True))
            Y_pred = Y_pred.reshape(-1, 1)
            sort_idx = np.argsort(Y_pred.flatten())
            Y_pred_sorted = Y_pred[sort_idx, :]
            Y_sorted = self.Y[sort_idx, :]
            Y_pred_binned = [np.mean(Y_pred_sorted[i:min(i+batch_size, self.n), :]) for i in np.arange(0, self.n, batch_size)]
            Y_binned = [np.mean(Y_sorted[i:min(i+batch_size, self.n), :]) for i in np.arange(0, self.n, batch_size)]
            plt.figure(figsize=(3, 3))
            plt.scatter(Y_pred_binned, Y_binned)
            plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), color="red", linestyle="--")
            plt.xlabel(r"$\hat{p}(Y_i = 1)$"); plt.ylabel(r"$Y$")
            plt.title("Binned residual plot")

        # plot the residuals vs features
        if "residual" in method:
            ncol = 4; nrow = (self.p + 1) // ncol + 1
            plt.figure(figsize=(3*ncol, 3*nrow))
            plt.subplot(nrow, ncol, 1)
            Y_pred_1 = self.logit.predict(sm.add_constant(self.X, prepend=True))
            Y_pred_1 = np.clip(Y_pred_1, 1e-5, 1-1e-5).flatten()
            Y_pred_0 = 1 - Y_pred_1
            Y_pred_logit = np.log(Y_pred_1 / Y_pred_0)

            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                plt.scatter(self.X[:, i], Y_pred_logit, s=1)
                smoother = self._smoother(self.X[:, i], Y_pred_logit, type=smoother_type)
                plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
                plt.legend()
                plt.xlabel(self.X_columns[i])
                plt.ylabel(r"$\log \frac{p_1}{p_0}$")
            plt.suptitle("Residuals vs features")
            plt.tight_layout()

        # partial residual plot
        if "partial_residual" in method:
            Y_pred = self.logit.predict(sm.add_constant(self.X, prepend=True))
            Y_pred = np.clip(Y_pred.flatten(), 1e-5, 1-1e-5)
            deviance = np.zeros(self.n); deviance[:] = np.nan
            deviance_residual = np.zeros(self.n); deviance_residual[:] = np.nan
            for i in range(self.n):
                deviance[i] = -2*np.log(Y_pred[i]) if self.Y[i, 0] == 1 else -2*np.log(1-Y_pred[i])
                deviance_residual[i] = np.sign(self.Y[i, 0] - Y_pred[i]) * np.sqrt(deviance[i])

            ncol = 3; nrow = int(np.ceil(self.p / ncol))
            plt.figure(figsize=(3*ncol, 3*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                if hasattr(self.logit, "params"):
                    partial_residual = deviance_residual + self.logit.params[i+1] * self.X[:, i]
                if hasattr(self.logit, "coef_"):
                    partial_residual = deviance_residual + self.logit.coef_[i+1] * self.X[:, i]
                plt.scatter(self.X[:, i], partial_residual, s=1)
                smoother = self._smoother(self.X[:, i], partial_residual, type=smoother_type)
                plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
                plt.xlabel(self.X_columns[i]); plt.ylabel(r"$\varepsilon^{dev} + \hat{\beta}_jx_{ij}$"); plt.legend()
            plt.suptitle("Partial residuals vs features")
            plt.tight_layout()

        # compare with polynomial model
        if ("interaction_term" in method) and (self.c == 2):
            LR_hist = np.zeros((self.p, self.p)); LR_hist[:] = np.nan
            AIC_hist = np.zeros((self.p, self.p)); AIC_hist[:] = np.nan
            BIC_hist = np.zeros((self.p, self.p)); BIC_hist[:] = np.nan
            CV_error = np.zeros((self.p, self.p)); CV_error[:] = np.nan

            logit = self._fit_logit(sm.add_constant(self.X, prepend=True), self.Y.flatten())
            if hasattr(logit, "llf"):
                LR_benchmark = (logit.llf, logit.df_model)

            logit = self._fit_logit(sm.add_constant(self.X_train, prepend=True), self.Y_train.flatten())
            Y_pred = logit.predict(sm.add_constant(self.X_test, prepend=True)).reshape((-1, 1))
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape((-1, 1))
            CV_error_benchmark = np.mean(np.abs(self.Y_test - Y_pred_binary))

            for i in range(self.p):
                for j in range(self.p):
                    X_temp = np.concatenate([self.X, (self.X[:, i]*self.X[:, j]).reshape((-1, 1))], axis=1)
                    logit = self._fit_logit(sm.add_constant(X_temp, prepend=True), self.Y.flatten())
                    if hasattr(logit, "llf"):
                        LR = 2*(logit.llf - LR_benchmark[0])
                        df_diff = logit.df_model - LR_benchmark[1]
                        LR_hist[i, j] = scipy.stats.chi2.sf(LR, df_diff)
                        AIC_hist[i, j] = logit.aic - self.logit.aic
                        BIC_hist[i, j] = logit.bic - self.logit.bic

                    X_temp = np.concatenate([self.X_train, (self.X_train[:, i]*self.X_train[:, j]).reshape((-1, 1))], axis=1)
                    logit = self._fit_logit(sm.add_constant(X_temp, prepend=True), self.Y_train.flatten())
                    X_test_temp = np.concatenate([self.X_test, (self.X_test[:, i]*self.X_test[:, j]).reshape((-1, 1))], axis=1)
                    Y_pred = logit.predict(sm.add_constant(X_test_temp, prepend=True)).reshape(-1, 1)
                    Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape((-1, 1))
                    CV_error_current = np.mean(np.abs(self.Y_test - Y_pred_binary))
                    CV_error[i, j] = (CV_error_current - CV_error_benchmark) / CV_error_benchmark

            self.nonlinearity_test["interaction_term_metric"] = [LR_hist, AIC_hist, BIC_hist, CV_error]

            plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            colors = ['red', 'yellow', 'green']
            bounds = [0, 0.01, 0.05, 1]
            cmap = matplotlib.colors.ListedColormap(colors)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            plt.imshow(LR_hist, cmap=cmap, norm=norm)
            plt.xticks(range(self.p), self.X_columns, rotation=90)
            plt.yticks(range(self.p), self.X_columns)
            plt.colorbar()
            plt.title(r"p-value of likelihood-ratio test")

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

    def seperation(self):
        '''
        Check for perfect seperation in the data.
        '''
        if not hasattr(self, "logit"):
            raise Exception("Fit the model first.")
        self.seperation_hist = []
        idx_1 = np.where(self.Y.flatten() == 1)[0]
        idx_0 = np.where(self.Y.flatten() == 0)[0]
        for j in range(self.p):
            x1 = self.X[idx_1, j]; x0 = self.X[idx_0, j]
            if np.min(x1) > np.max(x0) or np.min(x0) > np.max(x1):
                print("Perfect seperation in feature %s." % self.X_columns[j])
                self.seperation_hist.append(j)

        if len(self.seperation_hist) == 0:
            print("No perfect seperation found in the data.")
        return self.seperation_hist

    def outlier(self, threshold="strict", method=["pearson", "deviance", "leverage", "cook_dist", "dfbetas"], is_output=True):
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
        '''
        if not hasattr(self, "logit"):
            raise Exception("Fit the model first.")

        self.outlier_test = {}
        # Pearson residuals
        Y_pred = self.logit.predict(sm.add_constant(self.X, prepend=True))
        pearson_residuals = (self.Y.flatten() - Y_pred.flatten()) / np.sqrt(Y_pred.flatten() * (1 - Y_pred.flatten()))
        self.outlier_test["pearson_residuals"] = np.where(np.abs(pearson_residuals) > 2)[0]

        # Deviance residuals
        deviance_residuals = np.zeros(self.n); deviance_residuals[:] = np.nan
        for i in range(self.n):
            if self.Y[i, 0] == 1:
                deviance_residuals[i] = np.sign(self.Y[i, 0] - Y_pred[i]) * np.sqrt(-2 * np.log(Y_pred[i]))
            else:
                deviance_residuals[i] = np.sign(self.Y[i, 0] - Y_pred[i]) * np.sqrt(-2 * np.log(1 - Y_pred[i]))
        self.outlier_test["deviance_residuals"] = np.where(np.abs(deviance_residuals) > 2)[0]

        # leverage
        W = np.diag(Y_pred.flatten() * (1 - Y_pred.flatten()))
        H = np.diag(X.dot(np.linalg.inv(X.T.dot(W).dot(X))).dot(X.T))
        self.outlier_test["leverage"] = np.where(H > 2 * self.p / self.n)[0]

        # Cook's distance
        cook_dist = np.power(pearson_residuals, 2)*H/(self.p * np.power(1-H, 2))
        self.outlier_test["cooks_distance"] = np.where(cook_dist > 4/self.n)[0]

        # DFBETAS
        if "dfbetas" in method:
            self.outlier_test["dfbetas"] = []
            for i in range(self.n):
                X_temp = self.X[~(np.arange(self.n) == i), :]
                Y_temp = self.Y[~(np.arange(self.n) == i), :]
                logit = self._fit_logit(sm.add_constant(X_temp, prepend=True), Y_temp.flatten())
                if hasattr(logit, "params") and hasattr(logit, "bse"):
                    beta_diff = (self.logit.params[1:] - logit.params[1:])/self.logit.bse[1:]
                if any(np.abs(beta_diff) > 2/ np.sqrt(self.n)):
                    self.outlier_test["dfbetas"].append(i)
            self.outlier_test["dfbetas"] = np.array(self.outlier_test["dfbetas"])

        outlier_summary = np.zeros((self.n, 5))
        outlier_summary[self.outlier_test["pearson_residuals"], 0] = 1
        outlier_summary[self.outlier_test["deviance_residuals"], 1] = 1
        outlier_summary[self.outlier_test["leverage"], 2] = 1
        outlier_summary[self.outlier_test["cooks_distance"], 3] = 1
        if "dfbetas" in method:
            outlier_summary[self.outlier_test["dfbetas"], 4] = 1

        if is_output:
            plt.figure(figsize=(6, 8))
            plt.subplot(2, 1, 1)
            plt.scatter(pearson_residuals, H, s=1)
            x_min, x_max = plt.gca().get_xlim()
            y_min, y_max = plt.gca().get_ylim()
            plt.hlines(y=2*self.p/self.n, xmin=x_min, xmax=x_max, color='red', linestyle='--')
            plt.vlines(x=[-2, 2], ymin=y_min, ymax=y_max, color='red', linestyle='--')
            plt.fill_between([-2, 2], 0, 2*self.p/self.n, color="green", alpha=0.2, label="not outlier/influential obs.")
            plt.legend()
            plt.xlabel("external studentized residuals"); plt.ylabel("leverage")

            plt.subplot(4, 1, 3)
            plt.imshow(outlier_summary.T, cmap='Reds', vmin=0, vmax=1, aspect='auto')
            plt.colorbar()
            plt.yticks(range(5), ["stu. res.", "p-value", "leverage", "cooks dist.", "dfbetas"])
            plt.xlabel("observation index")

            plt.subplot(4, 1, 4)
            plt.scatter(range(self.n), np.sum(outlier_summary, axis=1), s=5, color="red")
            plt.xlim(0, self.n-1)
            plt.ylabel("Number of \n outlier/influential obs.")

            plt.suptitle("Detect Outlier/Influential Obs. Plot")
            plt.tight_layout()

        return self.outlier_test

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
        self.feature_selection_best_subset(criterion="accuracy (out-of-sample)", is_plot=is_plot)
        self.feature_selection_forward_stepwise(criterion="accuracy (out-of-sample)", is_plot=is_plot)
        self.feature_selection_ridge_lasso(is_plot=is_plot)

        if is_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_best_subset_summary[i][4] for i in range(1, self.p+1)], "-o", label="best subset")
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][4] for i in range(1, self.p+1)], "-o", label="forward stepwise")
            #plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_ridge_summary[i][1] for i in range(1, self.p+1)], "-o", label="ridge")
            plt.plot([i for i in np.arange(1, self.p+1, 1) if i in self.feature_selection_lasso_summary.keys()], [self.feature_selection_lasso_summary[i][1] for i in np.arange(1, self.p+1, 1) if i in self.feature_selection_lasso_summary.keys()], "-o", label="lasso")
            plt.hlines(y=self.feature_selection_best_subset_summary[0][4], xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1],
                       color='gray', linestyle='--', label="no feature")
            plt.xlabel("number of features")
            plt.ylabel("accuracy (out-of-sample)")
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
            plt.suptitle("feature selection")
            plt.tight_layout()

    def feature_selection_best_subset(self, criterion="accuracy (out-of-sample)", is_plot=True):
        '''
        Feature selection by best subset selection.
        params:
            criterion: str, "accuracy (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "accuracy (out-of-sample)"

        return:
            self.feature_selection_best_subset_summary: dict
                "feature_number": [feature_idx, accuracy (in-sample), AIC (in-sample), BIC (in-sample), accuracy (out-of-sample)]
        '''
        print("--- Feature selection by best subset ---")
        self.feature_selection_best_subset_summary = collections.defaultdict(list)
        self.feature_selection_best_subset_summary["feature_number"] = ["feature_idx", "accuracy (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "accuracy (out-of-sample)"]

        log = collections.defaultdict(list)
        logit = self._fit_logit(sm.add_constant(self.X, prepend=True)[:, [0]], self.Y.flatten(), disp=False)
        Y_pred = logit.predict(sm.add_constant(self.X, prepend=True)[:, [0]]).reshape(-1, 1)
        Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
        accuracy_is = 1 - np.mean(np.abs(self.Y - Y_pred_binary))

        logit_oos = self._fit_logit(sm.add_constant(self.X_train, prepend=True)[:, [0]], self.Y_train.flatten(), disp=False)
        Y_pred = logit_oos.predict(sm.add_constant(self.X_test, prepend=True)[:, [0]]).reshape(-1, 1)
        Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
        accuracy_oos = 1 - np.mean(np.abs(self.Y_test - Y_pred_binary))
        if hasattr(logit, "aic") and hasattr(logit, "bic"):
            log[0].append([[], accuracy_is, logit.aic, logit.bic, accuracy_oos])
        else:
            log[0].append([[], accuracy_is, np.nan, np.nan, accuracy_oos])

        for feature_num in range(1, self.p + 1):
            for feature_idx in itertools.combinations(range(0, self.p), feature_num):
                feature_idx = list(feature_idx)
                logit = self._fit_logit(sm.add_constant(self.X[:, feature_idx], prepend=True), self.Y.flatten(), disp=False)
                Y_pred = logit.predict(sm.add_constant(self.X[:, feature_idx], prepend=True)).reshape(-1, 1)
                Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                accuracy_is = 1- np.mean(np.abs(self.Y - Y_pred_binary))

                logit_oos = self._fit_logit(sm.add_constant(self.X_train[:, feature_idx], prepend=True), self.Y_train.flatten(), disp=False)
                Y_pred = logit_oos.predict(sm.add_constant(self.X_test[:, feature_idx], prepend=True)).reshape(-1, 1)
                Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                accuracy_oos = 1- np.mean(np.abs(self.Y_test - Y_pred_binary))

                if hasattr(logit, "aic") and hasattr(logit, "bic"):
                    log[feature_num].append([feature_idx, accuracy_is, logit.aic, logit.bic, accuracy_oos])
                else:
                    log[feature_num].append([feature_idx, accuracy_is, np.nan, np.nan, accuracy_oos])

        plt.figure(figsize=(8, 6))
        plt.subplot(4, 1, 1)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[1])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[1] for i in log[feature_num]], "o", color="gray")
            if criterion == "accuracy (in-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][1] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("accuracy (in-sample)")

        plt.subplot(4, 1, 2)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[2])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[2] for i in log[feature_num]], "o", color="gray")
            if criterion == "AIC (in-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][2] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("AIC (in-sample)")

        plt.subplot(4, 1, 3)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[3])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[3] for i in log[feature_num]], "o", color="gray")
            if criterion == "BIC (in-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][3] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("BIC (in-sample)")

        plt.subplot(4, 1, 4)
        for feature_num in np.arange(0, self.p + 1):
            log[feature_num].sort(reverse=True, key=lambda x: x[4])
            plt.plot([feature_num for _ in range(len(log[feature_num]))], [i[4] for i in log[feature_num]], "o", color="gray")
            if criterion == "accuracy (out-of-sample)":
                print("feature_num: %d, best feature: %s, accuracy (in-sample): %.4f, AIC (in-sample): %.4f, BIC (in-sample): %.4f, accuracy (out-of-sample): %.4f" % (feature_num, log[feature_num][0][0], log[feature_num][0][1], log[feature_num][0][2], log[feature_num][0][3], log[feature_num][0][4]))
                self.feature_selection_best_subset_summary[feature_num] = log[feature_num][0]
        plt.plot(np.arange(0, self.p + 1), [log[i][0][4] for i in np.arange(0, self.p+1, 1)], "o-", color="red")
        plt.xlabel("feature number"); plt.ylabel("accuracy (out-of-sample)")
        plt.suptitle("Feature selection by best subset selection with criterion: %s" % criterion)
        plt.tight_layout()

        if not is_plot:
            plt.close(plt.gcf())

        return self.feature_selection_best_subset_summary

    def feature_selection_forward_stepwise(self, criterion="accuracy (out-of-sample)", is_plot=True):
        '''
        Feature selection by forward stepwise selection.
        params:
            criterion: str, "accuracy (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "accuracy (out-of-sample)"
        
        return:
            self.feature_selection_forward_stepwise_summary: dict
                "feature_number": [feature_idx, accuracy (in-sample), AIC (in-sample), BIC (in-sample), accuracy (out-of-sample)]
        '''
        print("--- Feature selection by forward stepwise selection ---")
        self.feature_selection_forward_stepwise_summary = collections.defaultdict(list)
        self.feature_selection_forward_stepwise_summary["feature_number"] = ["feature_idx", "accuracy (in-sample)", "AIC (in-sample)", "BIC (in-sample)", "accuracy (out-of-sample)"]

        selected_feature = set()
        log = collections.defaultdict(list)
        logit = self._fit_logit(sm.add_constant(self.X, prepend=True)[:, [0]], self.Y.flatten(), disp=False)
        Y_pred = logit.predict(sm.add_constant(self.X, prepend=True)[:, [0]]).reshape(-1, 1)
        Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
        accuracy_is = 1 - np.mean(np.abs(self.Y - Y_pred_binary))

        logit_oos = self._fit_logit(sm.add_constant(self.X_train, prepend=True)[:, [0]], self.Y_train.flatten(), disp=False)
        Y_pred = logit_oos.predict(sm.add_constant(self.X_test, prepend=True)[:, [0]]).reshape(-1, 1)
        Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
        accuracy_oos = 1 - np.mean(np.abs(self.Y_test - Y_pred_binary))
        if hasattr(logit, "aic") and hasattr(logit, "bic"):
            log[0].append([[], accuracy_is, logit.aic, logit.bic, accuracy_oos])
        else:
            log[0].append([[], accuracy_is, np.nan, np.nan, accuracy_oos])

        for feature_num in np.arange(1, self.p+1, 1):
            for feature_idx in range(self.p):
                if feature_idx not in selected_feature:
                    logit = self._fit_logit(sm.add_constant(self.X[:, list(selected_feature) + [feature_idx]], prepend=True), self.Y)
                    Y_pred = logit.predict(sm.add_constant(self.X[:, list(selected_feature) + [feature_idx]], prepend=True)).reshape(-1, 1)
                    Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                    accuracy_is = 1 - np.mean(np.abs(self.Y - Y_pred_binary))

                    logit_oos = self._fit_logit(sm.add_constant(self.X_train[:, list(selected_feature) + [feature_idx]], prepend=True), self.Y_train)
                    Y_pred = logit_oos.predict(sm.add_constant(self.X_test[:, list(selected_feature) + [feature_idx]], prepend=True)).reshape(-1, 1)
                    Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                    accuracy_oos = 1 - np.mean(np.abs(self.Y_test - Y_pred_binary))
                    if hasattr(logit, "aic") and hasattr(logit, "bic"):
                        log[feature_num].append([list(selected_feature) + [feature_idx], accuracy_is, logit.aic, logit.bic, accuracy_oos])
                    else:
                        log[feature_num].append([list(selected_feature) + [feature_idx], accuracy_is, np.nan, np.nan, accuracy_oos])

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
        
        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(4,1,1)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][1] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("accuracy (in-sample)")
            plt.subplot(4,1,2)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][2] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("AIC (in-sample)")
            plt.subplot(4,1,3)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][3] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("BIC (in-sample)")
            plt.subplot(4,1,4)
            plt.plot(np.arange(1, self.p+1, 1), [self.feature_selection_forward_stepwise_summary[i][4] for i in np.arange(1, self.p+1, 1)], "o-")
            plt.xlabel("feature number"); plt.ylabel("accuracy (out-of-sample)")
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
        self.feature_selection_ridge_summary["feature_number"] = ["feature_idx", "accuracy (out-of-sample)"]
        alpha_list = np.logspace(-6, 6, num=10000, base=10)
        log = collections.defaultdict(list)

        for alpha in alpha_list:
            logit = sklearn.linear_model.LogisticRegression(penalty='l2', C=1/alpha, fit_intercept=True)
            logit.fit(self.X_train, self.Y_train.flatten())
            Y_pred = logit.predict(self.X_test).reshape(-1, 1)
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
            accuracy_oos = 1 - np.mean(np.abs(self.Y_test - Y_pred_binary))
            log[alpha] = [logit.coef_.T, logit.intercept_, accuracy_oos]

        critical_alpha = []
        for i in range(len(alpha_list)-1, 0, -1):
            alpha = alpha_list[i]
            coef_1 = log[alpha_list[i]][0].flatten(); coef_2 = log[alpha_list[i-1]][0].flatten()
            coef_1_nonzero_ct = len(np.where(np.abs(coef_1) > beta_threshold)[0])
            coef_2_nonzero_ct = len(np.where(np.abs(coef_2) > beta_threshold)[0])
            if coef_2_nonzero_ct > coef_1_nonzero_ct:
                self.feature_selection_ridge_summary[coef_1_nonzero_ct] = [list(np.where(np.abs(coef_1) > beta_threshold)[0]), log[alpha][2]]
                print("Ridge feature number: %d, alpha: %.4f, selected feature index: %s, accuracy (out-of-sample): %.4f" % (coef_1_nonzero_ct, alpha, self.feature_selection_ridge_summary[coef_1_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)
            if i == 1:
                self.feature_selection_ridge_summary[coef_2_nonzero_ct] = [list(np.where(np.abs(coef_2) > beta_threshold)[0]), log[alpha][2]]
                print("Ridge feature number: %d, alpha: %.4f, selected feature index: %s, accuracy (out-of-sample): %.4f" % (coef_2_nonzero_ct, alpha, self.feature_selection_ridge_summary[coef_2_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)

        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(2,1,1)
            plt.plot(alpha_list, [log[i][2] for i in alpha_list])
            plt.xscale("log"); plt.xlabel("alpha"); plt.ylabel("accuracy (out-of-sample)")
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
        self.feature_selection_lasso_summary["feature_number"] = ["feature_idx", "accuracy (out-of-sample)"]
        alpha_list = np.logspace(-6, 6, num=10000, base=10)
        log = collections.defaultdict(list)

        for alpha in alpha_list:
            logit = sklearn.linear_model.LogisticRegression(penalty='l1', C=1/alpha, fit_intercept=True, solver='liblinear')
            logit.fit(self.X_train, self.Y_train.flatten())
            Y_pred = logit.predict(self.X_test).reshape(-1, 1)
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
            accuracy_oos = 1 - np.mean(np.abs(self.Y_test - Y_pred_binary))
            log[alpha] = [logit.coef_.T.reshape((-1, 1)), logit.intercept_, accuracy_oos]

        critical_alpha = []
        for i in range(len(alpha_list)-1, 0, -1):
            alpha = alpha_list[i]
            coef_1 = log[alpha_list[i]][0].flatten(); coef_2 = log[alpha_list[i-1]][0].flatten()
            coef_1_nonzero_ct = len(np.where(np.abs(coef_1) > beta_threshold)[0])
            coef_2_nonzero_ct = len(np.where(np.abs(coef_2) > beta_threshold)[0])
            if coef_2_nonzero_ct > coef_1_nonzero_ct:
                self.feature_selection_lasso_summary[coef_1_nonzero_ct] = [list(np.where(np.abs(coef_1) > beta_threshold)[0]), log[alpha][2]]
                print("Lasso feature number: %d, alpha: %.4f, selected feature index: %s, accuracy (out-of-sample): %.4f" % (coef_1_nonzero_ct, alpha, self.feature_selection_lasso_summary[coef_1_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)
            if i == 1:
                self.feature_selection_lasso_summary[coef_2_nonzero_ct] = [list(np.where(np.abs(coef_2) > beta_threshold)[0]), log[alpha][2]]
                print("Lasso feature number: %d, alpha: %.4f, selected feature index: %s, accuracy (out-of-sample): %.4f" % (coef_2_nonzero_ct, alpha, self.feature_selection_lasso_summary[coef_2_nonzero_ct][0], log[alpha][2]))
                critical_alpha.append(alpha)

        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.subplot(2,1,1)
            plt.plot(alpha_list, [log[i][2] for i in alpha_list])
            plt.xscale("log"); plt.xlabel("alpha"); plt.ylabel("accuracy (out-of-sample)")
            plt.subplot(2,1,2)
            for feature_idx in range(self.p):
                plt.plot(alpha_list, [log[i][0][feature_idx, 0] for i in alpha_list],  label=self.X_columns[feature_idx])
            plt.axhline(y=0, color='black', linestyle='--')
            plt.vlines(critical_alpha, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1], color="red", linestyle="--", label="critical alpha")
            plt.xscale("log"); plt.xlabel("alpha"); plt.ylabel(r"$\beta$"); plt.legend()
            plt.suptitle("Feature selection by lasso regression")
            plt.tight_layout()

    def _fit_logit(self, X, Y, disp=False):
        '''
        Fit logistic regression model using various methods.
        If all methods fail, use sklearn's LogisticRegression with L2 regularization.

        params:
            X: np.ndarray, shape (n_samples, n_features).
                Input data with intercept, consistent with statsmodels' default.
            Y: np.ndarray, shape (n_samples), target data
            disp: bool, whether to print the fitting process

        return:
            logit: fitted logistic regression model

        comments:
            To predict, we need to add a constant term to X before using logit.predict().
        '''
        Y = Y.flatten()
        logit = None
        fit_method = ["newton", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]
        for method in fit_method:
            try:
                logit = statsmodels.api.Logit(Y, X).fit(disp=False, method=method)
                if disp:
                    print("Logistic regression converged with method: %s" % method)
                break
            except:
                if disp:
                    print("Logistic regression failed to converge with method: %s" % method)
        if logit is None:
            logit = sklearn.linear_model.LogisticRegression(penalty='l2', C=1.0, fit_intercept=False)
            logit.fit(X, Y.flatten())
        return logit

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

'''
is_binary = True
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/vowel_train.csv"), index_col=0)
X = data.iloc[:, 1:].to_numpy()
Y = data.iloc[:, 0].to_numpy().reshape(-1, 1) - 1
if is_binary:
    idx = np.where(Y.flatten() <= 1)[0]
    X = X[idx, :]; Y = Y[idx, :]
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/vowel_test.csv"), index_col=0)
X_test = data.iloc[:, 1:].to_numpy()
Y_test = data.iloc[:, 0].to_numpy().reshape(-1, 1) - 1
if is_binary:
    idx = np.where(Y_test.flatten() <= 1)[0]
    X_test = X_test[idx, :]; Y_test = Y_test[idx, :]
'''

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_from_ESL/south_african_heart_disease.csv"), index_col=0)
data["famhist"] = data["famhist"].map({"Present": 1, "Absent": 0})
X = data.iloc[:, 0:(data.shape[1]-1)].to_numpy()
Y = data.iloc[:, -1].to_numpy().reshape(-1, 1)

model = logistic_regression_binary(X, Y, X_test=None, Y_test=None, X_columns=data.columns[0:(data.shape[1]-1)], is_normalize=True, test_size_ratio=0.2)
model.fit(is_output=False)
#_ = model.predict(X_test)
#model.visualize_data()
#_ = model.nonlinearity()
#_ = model.colinearity()
#_ = model.seperation()
#_ = model.outlier(threshold="strict", is_output=True)
#_ = model.feature_selection_best_subset()
#_ = model.feature_selection_forward_stepwise(criterion="accuracy (out-of-sample)", is_plot=True)
#model.feature_selection_ridge_lasso(beta_threshold=1e-3, is_plot=True)
#model.feature_selection_all(is_plot=True)



