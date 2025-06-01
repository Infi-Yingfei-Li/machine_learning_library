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
        if self.c == 2:
            self.logit = self._fit_logit(self.X, self.Y, disp=is_output)
            if is_output and hasattr(self.logit, "aic"):
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
        else:
            print("Warning: Multi-class logistic regression has not been implemented rigorously yet. The results may not be reliable.")
            self.logit = statsmodels.api.MNLogit(self.Y.flatten(), sm.add_constant(self.X, prepend=True)).fit()
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

    def predict(self, X):
        '''
        Predict the probability of class for the given input features.
        If the model is binary, return the probability of class 1.
        If the model is multi-class, return the probability of each class.
        params:
            X: feature matrix, shape (n, p)
        return:
            Y_pred: predicted target values, shape (n, c)
        '''
        if not hasattr(self, "logit"):
            raise Exception("Fit the model first.")
        if self.is_normalize:
            X = (X - self.X_mean) / self.X_std
        Y_pred = self.logit.predict(sm.add_constant(X, prepend=True))
        if self.c == 2:
            Y_pred = Y_pred.reshape(-1, 1)
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

    def nonlinearity(self, class_=1, smoother_type="polynomial", method=["binned_residual", "residual", "partial_residual", "interaction_term"]):
        '''
        Diagonostic analysis on non-linearity. 
        For binary classification, this function detects potential nonlinearity for logit(p_1/(1-p_1)).
        For multi-class classification, this function detects potential nonlinearity for logit(p_i/p_0).

        Analysis include:
            (1) binned residual plot
                Sort the predicted probabilities and group them into bins. 
                Plot the average predicted probabilities vs the average observed values in each bin.
                For binary classification, the plot should be close to a 45-degree line.

            (2) residual versus features plot
                Plot the estimated logit(p_i/p_0) vs each feature. Curvature in the plot suggests nonlinearity.

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
                    AIC, BIC, R^2 (out of sample): accept large model when we observe a significant reduction
        
        params:
            class_: class to check for nonlinearity. Default is 1. Class 0 is the reference class.
            smoother_type: type for smoother in the plots. See self._smoother for details.

        returns:
            self.nonlinearity_test: dict
                - "interaction_term_metric" -- [p-value of F-statistics, AIC, BIC, out-of-sample R^2]
        '''
        if not hasattr(self, "logit"):
            raise Exception("Fit the model first.")
        self.nonlinearity_test = {}

        # binned residual plot
        if "binned_residual" in method:
            bins = 20; batch_size = int(np.ceil(self.n / bins))
            if self.c == 2:
                Y_pred = self.logit.predict(sm.add_constant(self.X, prepend=True)).reshape(-1, 1)
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
            else:
                Y_pred = self.logit.predict(sm.add_constant(self.X, prepend=True))[:, class_]
                Y_pred = Y_pred.reshape(-1, 1)
                sort_idx = np.argsort(Y_pred.flatten())
                Y_pred_sorted = Y_pred[sort_idx, :]
                Y_sorted = self.Y[sort_idx, :]
                Y_pred_binned = [np.mean(Y_pred_sorted[i:min(i+batch_size, self.n), :]) for i in np.arange(0, self.n, batch_size)]
                Y_binned = [np.mean(Y_sorted[i:min(i+batch_size, self.n), :]) for i in np.arange(0, self.n, batch_size)]
                plt.figure(figsize=(3, 3))
                plt.scatter(Y_pred_binned, Y_binned)
                plt.hlines(class_, 0, 1, color="red", linestyle="--")
                plt.vlines(1, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color="red", linestyle="--")
                plt.xlabel(r"$\hat{p}(Y_i = $"+str(class_)+")"); plt.ylabel(r"$Y$")
                plt.suptitle("Binned residual plot")
                plt.tight_layout()

        # plot the residuals vs features
        if "residual" in method:
            ncol = 4; nrow = (self.p + 1) // ncol + 1
            plt.figure(figsize=(3*ncol, 3*nrow))
            plt.subplot(nrow, ncol, 1)
            if self.c == 2:
                Y_pred_1 = np.clip(self.logit.predict(sm.add_constant(self.X, prepend=True)).flatten(), 1e-5, 1-1e-5)
            else:
                Y_pred_1 = np.clip(self.logit.predict(sm.add_constant(self.X, prepend=True))[:, class_], 1e-5, 1-1e-5)
            Y_pred_0 = 1 - Y_pred_1
            Y_pred_logit = np.log(Y_pred_1 / Y_pred_0)

            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                plt.scatter(self.X[:, i], Y_pred_logit, s=1)
                smoother = self._smoother(self.X[:, i], Y_pred_logit, type=smoother_type)
                plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
                plt.legend()
                plt.xlabel(self.X_columns[i])
                if self.c == 2:
                    plt.ylabel(r"$\log \frac{p_1}{p_0}$")
                else:
                    plt.ylabel(r"$\log \frac{p_c}{p_0}$")
            plt.suptitle("Residuals vs features")
            plt.tight_layout()

        # partial residual plot
        if "partial_residual" in method:
            if self.c == 2:
                Y_pred = self.logit.predict(sm.add_constant(self.X, prepend=True)).flatten()
                Y_pred = np.clip(Y_pred, 1e-5, 1-1e-5)
                deviance = np.zeros(self.n); deviance[:] = np.nan
                deviance_residual = np.zeros(self.n); deviance_residual[:] = np.nan
                for i in range(self.n):
                    deviance[i] = -2*np.log(Y_pred[i]) if self.Y[i, 0] == 1 else -2*np.log(1-Y_pred[i])
                    deviance_residual[i] = np.sign(self.Y[i, 0] - Y_pred[i]) * np.sqrt(deviance[i])
            else:
                Y_pred = self.logit.predict(sm.add_constant(self.X, prepend=True))
                deviance = np.zeros(self.n); deviance[:] = np.nan
                for i in range(self.n):
                    deviance[i] = -2*np.log(Y_pred[i, self.Y[i, 0]])
                deviance_residual = np.sqrt(deviance)

            ncol = 3; nrow = int(np.ceil(self.p / ncol))
            plt.figure(figsize=(3*ncol, 3*nrow))
            for i in range(self.p):
                plt.subplot(nrow, ncol, i + 1)
                if self.c == 2:
                    partial_residual = deviance_residual + self.logit.params[i+1] * self.X[:, i]
                else:
                    partial_residual = deviance_residual + self.logit.params[i+1, class_] * self.X[:, i]

                plt.scatter(self.X[:, i], partial_residual, s=1)
                smoother = self._smoother(self.X[:, i], partial_residual, type=smoother_type)
                plt.plot(smoother["fit_x"], smoother["fit_y"], color="red", linestyle="--", label = f"$R^2$={smoother['R2']:.2f}")
                plt.xlabel(self.X_columns[i]); plt.ylabel(r"$\varepsilon^{dev} + \hat{\beta}_jx_{ij}$"); plt.legend()
            plt.suptitle("Partial residuals vs features for class {}".format(class_))
            plt.tight_layout()

        # compare with polynomial model
        if ("interaction_term" in method) and (self.c == 2):
            LR_hist = np.zeros((self.p, self.p)); LR_hist[:] = np.nan
            AIC_hist = np.zeros((self.p, self.p)); AIC_hist[:] = np.nan
            BIC_hist = np.zeros((self.p, self.p)); BIC_hist[:] = np.nan
            CV_error = np.zeros((self.p, self.p)); CV_error[:] = np.nan

            logit = self._fit_logit(self.X, self.Y.flatten())
            if hasattr(logit, "llf"):
                LR_benchmark = (logit.llf, logit.df_model)

            logit = self._fit_logit(self.X_train, self.Y_train.flatten())
            Y_pred = logit.predict(sm.add_constant(self.X_test, prepend=True)).reshape((-1, 1))
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape((-1, 1))
            CV_error_benchmark = np.mean(np.abs(self.Y_test - Y_pred_binary))

            for i in range(self.p):
                for j in range(self.p):
                    X_temp = np.concatenate([self.X, (self.X[:, i]*self.X[:, j]).reshape((-1, 1))], axis=1)
                    logit = self._fit_logit(X_temp, self.Y.flatten())
                    if hasattr(logit, "llf"):
                        LR = 2*(logit.llf - LR_benchmark[0])
                        df_diff = logit.df_model - LR_benchmark[1]
                        LR_hist[i, j] = scipy.stats.chi2.sf(LR, df_diff)
                        AIC_hist[i, j] = logit.aic - self.logit.aic
                        BIC_hist[i, j] = logit.bic - self.logit.bic

                    X_temp = np.concatenate([self.X_train, (self.X_train[:, i]*self.X_train[:, j]).reshape((-1, 1))], axis=1)
                    logit = self._fit_logit(X_temp, self.Y_train.flatten())
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

        if ("interaction_term" in method) and (self.c > 2):
            if class_ == 0:
                raise Exception("Class 0 is the base class for multi-class logistic regression.")
            LR_hist = np.zeros((self.p, self.p)); LR_hist[:] = np.nan
            AIC_hist = np.zeros((self.p, self.p)); AIC_hist[:] = np.nan
            BIC_hist = np.zeros((self.p, self.p)); BIC_hist[:] = np.nan
            CV_error = np.zeros((self.p, self.p)); CV_error[:] = np.nan

            obs_idx = np.where((self.Y.flatten() == class_) | (self.Y.flatten() == 0))[0]
            X_subclass = self.X[obs_idx, :]
            Y_subclass = self.Y[obs_idx, :].reshape(-1, 1)
            Y_subclass = (Y_subclass.flatten() == class_).astype(int).reshape(-1, 1)

            obs_idx_train = np.where((self.Y_train.flatten() == class_) | (self.Y_train.flatten() == 0))[0]
            X_train_subclass = self.X_train[obs_idx_train, :]
            Y_train_subclass = self.Y_train[obs_idx_train, :].reshape(-1, 1)
            Y_train_subclass = (Y_train_subclass.flatten() == class_).astype(int).reshape(-1, 1)

            obs_idx_test = np.where((self.Y_test.flatten() == class_) | (self.Y_test.flatten() == 0))[0]
            X_test_subclass = self.X_test[obs_idx_test, :]
            Y_test_subclass = self.Y_test[obs_idx_test, :].reshape(-1, 1)
            Y_test_subclass = (Y_test_subclass.flatten() == class_).astype(int).reshape(-1, 1)

            logit= self._fit_logit(X_subclass, Y_subclass.flatten())
            if hasattr(logit, "llf"):
                LR_benchmark = (logit.llf, logit.df_model)

            logit = self._fit_logit(X_train_subclass, Y_train_subclass.flatten())
            Y_pred = logit.predict(sm.add_constant(X_test_subclass, prepend=True)).reshape((-1, 1))
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape((-1, 1))
            CV_error_benchmark = np.mean(np.abs(Y_test_subclass - Y_pred_binary))

            for i in range(self.p):
                for j in range(self.p):
                    X_temp = np.concatenate([X_subclass, (X_subclass[:, i]*X_subclass[:, j]).reshape((-1, 1))], axis=1)
                    logit = self._fit_logit(X_temp, Y_subclass.flatten())
                    if hasattr(logit, "llf"):
                        LR = 2*(logit.llf - LR_benchmark[0])
                        df_diff = logit.df_model - LR_benchmark[1]
                        LR_hist[i, j] = scipy.stats.chi2.sf(LR, df_diff)
                        AIC_hist[i, j] = logit.aic - self.logit.aic
                        BIC_hist[i, j] = logit.bic - self.logit.bic

                    X_temp = np.concatenate([X_train_subclass, (X_train_subclass[:, i]*X_train_subclass[:, j]).reshape((-1, 1))], axis=1)
                    logit= self._fit_logit(X_temp, Y_train_subclass.flatten())
                    X_test_temp = np.concatenate([X_test_subclass, (X_test_subclass[:, i]*X_test_subclass[:, j]).reshape((-1, 1))], axis=1)
                    Y_pred = logit.predict(sm.add_constant(X_test_temp, prepend=True)).reshape(-1, 1)
                    Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                    CV_error_current = np.mean(np.abs(Y_test_subclass - Y_pred_binary))
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
            plt.suptitle("Polynomial model comparison for class {}".format(class_))
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
        if self.c == 2:
            idx_1 = np.where(self.Y.flatten() == 1)[0]
            idx_0 = np.where(self.Y.flatten() == 0)[0]
            for j in range(self.p):
                x1 = self.X[idx_1, j]; x0 = self.X[idx_0, j]
                if np.min(x1) > np.max(x0) or np.min(x0) > np.max(x1):
                    print("Perfect seperation in feature %s." % self.X_columns[j])
                    self.seperation_hist.append(j)
        else:
            for c in range(1, self.c):
                idx_c = np.where(self.Y.flatten() == c)[0]
                idx_0 = np.where(self.Y.flatten() == 0)[0]
                for j in range(self.p):
                    xc = self.X[idx_c, j]; x0 = self.X[idx_0, j]
                    if np.min(xc) > np.max(x0) or np.min(x0) > np.max(xc):
                        print("Perfect seperation in feature %s for class %d." % (self.X_columns[j], c))
                        self.seperation_hist.append((c, j))
        
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
        if self.c != 2:
            raise Exception("Outlier detection is currently only implemented for binary logistic regression.")

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
                logit = self._fit_logit(X_temp, Y_temp.flatten(), disp=False)
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
            plt.fill_between([-2, 2], 0, 2*self.p/self.n, color="green", alpha=0.2, label="not outlier")
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

        if self.c == 2:
            log = collections.defaultdict(list)
            logit = self._fit_logit(np.zeros((self.n, 0)), self.Y.flatten(), disp=False)
            Y_pred = logit.predict(np.ones((self.n, 1))).reshape(-1, 1)
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
            accuracy_is = 1 - np.mean(np.abs(self.Y - Y_pred_binary))

            logit_oos = self._fit_logit(np.zeros((self.Y_train.shape[0], 0)), self.Y_train.flatten(), disp=False)
            Y_pred = logit_oos.predict(np.ones((self.Y_test.shape[0], 1))).reshape(-1, 1)
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
            accuracy_oos = 1 - np.mean(np.abs(self.Y_test - Y_pred_binary))
            if hasattr(logit, "aic") and hasattr(logit, "bic"):
                log[0].append([[], accuracy_is, logit.aic, logit.bic, accuracy_oos])
            else:
                log[0].append([[], accuracy_is, np.nan, np.nan, accuracy_oos])

            for feature_num in range(1, self.p + 1):
                for feature_idx in itertools.combinations(range(0, self.p), feature_num):
                    feature_idx = list(feature_idx)
                    logit = self._fit_logit(self.X[:, feature_idx], self.Y.flatten(), disp=False)
                    Y_pred = logit.predict(sm.add_constant(self.X[:, feature_idx], prepend=True)).reshape(-1, 1)
                    Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                    accuracy_is = 1- np.mean(np.abs(self.Y - Y_pred_binary))

                    logit_oos = self._fit_logit(self.X_train[:, feature_idx], self.Y_train.flatten(), disp=False)
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

        if self.c > 2:
            raise Exception("Feature selection by best subset is currently only implemented for binary logistic regression.")

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

        if self.c == 2:
            selected_feature = set()
            log = collections.defaultdict(list)
            logit = self._fit_logit(np.zeros((self.n, 0)), self.Y.flatten(), disp=False)
            Y_pred = logit.predict(np.ones((self.n, 1))).reshape(-1, 1)
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
            accuracy_is = 1 - np.mean(np.abs(self.Y - Y_pred_binary))

            logit_oos = self._fit_logit(np.zeros((self.Y_train.shape[0], 0)), self.Y_train.flatten(), disp=False)
            Y_pred = logit_oos.predict(np.ones((self.Y_test.shape[0], 1))).reshape(-1, 1)
            Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
            accuracy_oos = 1 - np.mean(np.abs(self.Y_test - Y_pred_binary))
            if hasattr(logit, "aic") and hasattr(logit, "bic"):
                log[0].append([[], accuracy_is, logit.aic, logit.bic, accuracy_oos])
            else:
                log[0].append([[], accuracy_is, np.nan, np.nan, accuracy_oos])

            for feature_num in np.arange(1, self.p+1, 1):
                for feature_idx in range(self.p):
                    if feature_idx not in selected_feature:
                        logit = self._fit_logit(self.X[:, list(selected_feature) + [feature_idx]], self.Y)
                        Y_pred = logit.predict(sm.add_constant(self.X[:, list(selected_feature) + [feature_idx]], prepend=True)).reshape(-1, 1)
                        Y_pred_binary = (Y_pred.flatten() >= 0.5).astype(int).reshape(-1, 1)
                        accuracy_is = 1 - np.mean(np.abs(self.Y - Y_pred_binary))

                        logit_oos = self._fit_logit(self.X_train[:, list(selected_feature) + [feature_idx]], self.Y_train)
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
        
        if self.c > 2:
            raise Exception("Feature selection by forward stepwise is currently only implemented for binary logistic regression.")

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
                Input data without intercept because the intercept is automatically added in the function.
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
                logit = statsmodels.api.Logit(Y, sm.add_constant(X, prepend=True)).fit(disp=False, method=method)
                if disp:
                    print("Logistic regression converged with method: %s" % method)
                break
            except:
                pass
        if logit is None:
            logit = sklearn.linear_model.LogisticRegression(penalty='l2', C=1.0, fit_intercept=False)
            logit.fit(sm.add_constant(X, prepend=True), Y.flatten())
        
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
