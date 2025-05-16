#%%
import os, datetime, tqdm, pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
# Suppress ConvergenceWarning
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

#%%
df1 = pd.read_csv("df.1.csv.gz", compression='gzip')
df2 = pd.read_csv("df.2.csv.gz", compression='gzip')
df3 = pd.read_csv("df.3.csv.gz", compression='gzip')
secm1 = pd.read_csv("secm.1.csv.gz", compression='gzip')
secm2 = pd.read_csv("secm.2.csv.gz", compression='gzip')
df = pd.concat([df1, df2, df3], axis=0)
secm = pd.concat([secm1, secm2], axis=0)
data = pd.merge(df, secm, on=["date_dt", "id"])

#%%
# data processing
id = data["id"].unique()
time_axis_str = data["date_dt"].unique()
time_axis = [datetime.datetime.strptime(j, "%Y-%m-%d") for j in time_axis_str]

feature = np.zeros((len(id), len(time_axis_str), 7)); feature[:] = np.nan
target = np.zeros((len(id), len(time_axis_str))); target[:] = np.nan
adjust_close = np.zeros((len(id), len(time_axis_str))); adjust_close[:] = np.nan
market_cap = np.zeros((len(id), len(time_axis_str))); market_cap[:] = np.nan

if os.path.exists("data.npz"):
    data = np.load("data.npz", allow_pickle=True)
    feature = data["feature"]; target = data["target"]; adjust_close = data["adjust_close"]; market_cap = data["market_cap"]

else:
    for i in tqdm.tqdm(range(len(id))):
        subdf = data[data["id"]==id[i]].copy()
        subdf = subdf.sort_values(by="date_dt")
        subdf_time_axis = [datetime.datetime.strptime(j, "%Y-%m-%d") for j in subdf["date_dt"]]
        for j in range(len(subdf_time_axis)):
            idx = np.searchsorted(time_axis, subdf_time_axis[j])
            feature[i, idx, :] = subdf[["x1", "x2", "x3", "x4", "x5", "x6", "x7"]].iloc[j, :].to_numpy() 
            target[i, idx] = subdf["y"].iloc[j]
            adjust_close[i, idx] = subdf["adjustedClose"].iloc[j]
            market_cap[i, idx] = subdf["marketcap"].iloc[j]

    np.savez_compressed("data.npz", feature=feature, target=target, adjust_close=adjust_close, market_cap=market_cap)

daily_return = np.zeros(adjust_close.shape); daily_return[:] = np.nan
for i in range(1, adjust_close.shape[1], 1):
    daily_return[:, i] = (adjust_close[:, i] - adjust_close[:, i-1])/(adjust_close[:, i-1])

market_cap_frac = market_cap/np.nansum(market_cap, axis=0)


#%%
# overview of  time series
# test stationarity of features
feature_number = 7
if os.path.exists("adfuller_result.npz"):
    data = np.load("adfuller_result.npz")
    adfuller_result = data["adfuller_result"]
else:
    adfuller_result = np.zeros((len(id), feature_number)); adfuller_result[:] = np.nan
    for i in tqdm.tqdm(range(len(id))):
        for j in range(feature_number):
            idx = np.where(~np.isnan(feature[i, :, j]))[0]
            try:
                adfuller_result[i, j] = adfuller(feature[i, :, j][idx], autolag='AIC')[1]
            except:
                pass

    np.savez_compressed("adfuller_result.npz", adfuller_result=adfuller_result)

plt.figure()
plt.hist(adfuller_result.flatten(), density=True, bins=100, range=(0, 1))
plt.xlabel("p-value"); plt.ylabel("density"); plt.title("p-value distribution of ADF test")
plt.yscale("log")

'''
Conclusion:
The p-value of ADF test is less than 0.05 for most of the features, which means most of the features are stationary.
'''

#%%
# test moving average and autoregression order of features
look_back_window=30
moving_average_order = np.zeros((len(id), feature_number, look_back_window)); moving_average_order[:] = np.nan
autoregression_order = np.zeros((len(id), feature_number, look_back_window)); autoregression_order[:] = np.nan
for i in tqdm.tqdm(range(len(id))):
    for j in range(feature_number):
        idx = np.where(~np.isnan(feature[i, :, j]))[0]
        try:
            moving_average_order[i, j, :] = acf(feature[i, :, j][idx], nlags=look_back_window)
        except:
            pass
        try:
            autoregression_order[i, j, :] = pacf(feature[i, :, j][idx], nlags=look_back_window)[0:look_back_window]
        except:
            pass

plt.figure()
for i in range(feature_number):
    plt.subplot(4,2,i+1)
    plt.plot(np.nanmean(moving_average_order[:, i, :], axis=0), label="feature {}".format(i+1))
    plt.fill_between(np.arange(look_back_window), np.nanmean(moving_average_order[:, i, :], axis=0)-np.nanstd(moving_average_order[:, i, :], axis=0), np.nanmean(moving_average_order[:, i, :], axis=0)+np.nanstd(moving_average_order[:, i, :], axis=0), alpha=0.5)
    plt.legend()
plt.suptitle("moving average order")
plt.tight_layout()

plt.figure()
for i in range(feature_number):
    plt.subplot(4,2,i+1)
    plt.plot(np.nanmean(autoregression_order[:, i, :], axis=0), label="feature {}".format(i+1))
    plt.fill_between(np.arange(look_back_window), np.nanmean(autoregression_order[:, i, :], axis=0)-np.nanstd(autoregression_order[:, i, :], axis=0), np.nanmean(autoregression_order[:, i, :], axis=0)+np.nanstd(autoregression_order[:, i, :], axis=0), alpha=0.5)
    plt.legend()
plt.suptitle("autoregression order")
plt.tight_layout()

'''
Conclusion:
no significant moving average and autoregression order detected by calculating autorcorrelation and partial autocorrelation
'''

#%%
train_idx = np.arange(0, int(0.8*len(time_axis)), 1)
test_idx = np.arange(int(0.8*len(time_axis)), len(time_axis), 1)

def generate_data(stock_idx, t_idx, look_back_window):
    ar = np.concatenate([feature[stock_idx, t_idx-look_back_window+1:t_idx+1, :].reshape((-1, 7)), daily_return[stock_idx, t_idx-look_back_window+1:t_idx+1].reshape((-1, 1)), market_cap_frac[stock_idx, t_idx-look_back_window+1:t_idx+1].reshape((-1, 1))], axis=1)
    ar2 = target[stock_idx, t_idx+1]
    if (~(np.isnan(ar).any())) and (~(np.isnan(ar2).any())):
        return ar.flatten().reshape((1, -1)), ar2
    else:
        return None, None



#%%
'''
Model 1: Lasso linear regression for feature selection, where the coefficient are independent of stock and time.
Steps:
(1) Construct features by varing the looking-back window;
(2) use Lasso regression to select effective features;
(3) evaluate the performance of the model by out-of-sample error and select optimal alpha that gives minimum out-of-sample error;
(4) Choose the optimal looking-back window by comparing the out-of-sample error.
'''
look_back_window_list = [1, 2, 3, 4, 5, 10, 30]
if os.path.exists("model1_oos_performance.pkl"):
    with open("model1_oos_performance.pkl", "rb") as f:
        model1_oos_performance = pickle.load(f)
else:
    model1_oos_performance = []
    def model1(look_back_window):
        print(look_back_window)
        # train sample
        X = []; Y = []
        for stock_idx in range(len(id)):
            for t_idx in range(look_back_window-1, (train_idx[-1])+1, 1):
                ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
                if ar is not None:
                    X.append(ar); Y.append(ar2)
        X = np.concatenate(X, axis=0); Y = np.array(Y)

        # test sample
        test_X = []; test_Y = []
        for stock_idx in range(len(id)):
            for t_idx in range(test_idx[0], test_idx[-1], 1):
                ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
                if ar is not None:
                    test_X.append(ar); test_Y.append(ar2)
        test_X = np.concatenate(test_X, axis=0); test_Y = np.array(test_Y)

        alpha_hist = []
        oos_error = []

        alpha = 1e-9
        while True:
            lasso = Lasso(alpha=alpha)
            lasso.fit(X, Y)
            lasso.predict(test_X)
            feature_number = len(np.where(np.abs(lasso.coef_)>1e-5)[0])
            if feature_number == 0:
                break
            alpha_hist.append(alpha)
            oos_error.append(np.mean(np.abs(lasso.predict(test_X)-test_Y)))
            alpha *= 2

        return alpha_hist, oos_error

    for look_back_window in tqdm.tqdm(look_back_window_list):
        alpha_hist, oos_error = model1(look_back_window)
        model1_oos_performance.append({"look_back_window": look_back_window, "alpha_hist": alpha_hist, "oos_error": oos_error})

    with open("model1_oos_performance.pkl", "wb") as f:
        pickle.dump(model1_oos_performance, f)

#plt.figure()
#plt.plot(look_back_window_list, [np.min(j["oos_error"]) for j in model1_oos_performance], marker="o")
#plt.xlabel("look back window"); plt.ylabel("out-of-sample error")

#%%
'''
Model 2: Lasso linear regression for feature selection, where the coefficient are dependent of stock and independent of time.
Steps:
(1) Construct features by varing the looking-back window;
(2) Sse Lasso regression to select effective features;
(3) Evaluate the performance of the model by out-of-sample error and select optimal alpha that gives minimum out-of-sample error;
(4) Choose the optimal looking-back window by comparing the out-of-sample error.
'''

def model2(look_back_window):
    print("look_back_window: ", look_back_window)
    alpha_hist_all = []; oos_error_all = []
    for stock_idx in tqdm.tqdm(range(len(id))):
        # train sample
        X = []; Y = []
        for t_idx in range(look_back_window-1, (train_idx[-1])+1, 1):
            ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
            if ar is not None:
                X.append(ar); Y.append(ar2)
        if len(X) == 0 or len(Y) == 0:
            continue
        X = np.concatenate(X, axis=0); Y = np.array(Y)

        # test sample
        test_X = []; test_Y = []
        for t_idx in range(test_idx[0], test_idx[-1], 1):
            ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
            if ar is not None:
                test_X.append(ar); test_Y.append(ar2)
        if len(test_X) == 0 or len(test_Y) == 0:
            continue
        test_X = np.concatenate(test_X, axis=0); test_Y = np.array(test_Y)

        alpha_hist = []
        oos_error = []

        alpha = 1e-9
        while True:
            lasso = Lasso(alpha=alpha)
            lasso.fit(X, Y)
            lasso.predict(test_X)
            feature_number = len(np.where(np.abs(lasso.coef_)>1e-5)[0])
            if feature_number == 0:
                break
            alpha_hist.append(alpha)
            oos_error.append(np.mean(np.abs(lasso.predict(test_X)-test_Y)))
            alpha *= 2

        if len(alpha_hist) == 0:
            continue
        alpha_hist_all.append(alpha_hist[np.argmin(oos_error)])
        oos_error_all.append(np.min(oos_error))

    return alpha_hist_all, oos_error_all

if os.path.exists("model2_oos_performance.pkl"):
    with open("model2_oos_performance.pkl", "rb") as f:
        model2_oos_performance = pickle.load(f)
else:
    look_back_window_list = [1, 2, 3, 4, 5, 10, 30]
    model2_oos_performance = []
    for look_back_window in look_back_window_list:
        alpha_hist_all, oos_error_all = model2(look_back_window)
        model2_oos_performance.append({"look_back_window": look_back_window, "alpha_hist": alpha_hist_all, "oos_error": oos_error_all})

    with open("model2_oos_performance.pkl", "wb") as f:
        pickle.dump(model2_oos_performance, f)

#%%
'''
Model 3: principal component regression (PCR) for feature selection, where the coefficient are independent of stock and time.
Steps:
(1) Construct features by varing the looking-back window;
(2) Calculate the principal components of the features;
(3) Varying the number of principal components to select effective features and perform linear regression;
(4) Evaluate the performance of the model by out-of-sample error and select optimal number of principal features that gives minimum out-of-sample error;
(5) Choose the optimal looking-back window by comparing the out-of-sample error.
'''

def model3(look_back_window):
    print(look_back_window)
    # train sample
    X = []; Y = []
    for stock_idx in range(len(id)):
        for t_idx in range(look_back_window-1, (train_idx[-1])+1, 1):
            ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
            if ar is not None:
                X.append(ar); Y.append(ar2)
    X = np.concatenate(X, axis=0); Y = np.array(Y)

    # test sample
    test_X = []; test_Y = []
    for stock_idx in range(len(id)):
        for t_idx in range(test_idx[0], test_idx[-1], 1):
            ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
            if ar is not None:
                test_X.append(ar); test_Y.append(ar2)
    test_X = np.concatenate(test_X, axis=0); test_Y = np.array(test_Y)

    oos_error = []
    for selected_feature_number in tqdm.tqdm(range(1, X.shape[1])):
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        X_reduced = U.dot(np.diag(S))[:, :selected_feature_number].reshape((-1, selected_feature_number))
        linreg = LinearRegression().fit(X_reduced, Y)
        coef = VT[:selected_feature_number, :].reshape((selected_feature_number, -1)).T.dot(linreg.coef_)
        intercept = linreg.intercept_
        predict_Y = test_X.dot(coef) + intercept

        oos_error.append(np.mean(np.abs(predict_Y-test_Y)))

    return list(range(1, X.shape[1])), oos_error

look_back_window_list = [1, 2, 3, 4, 5, 10, 30]
if os.path.exists("model3_oos_performance.pkl"):
    with open("model3_oos_performance.pkl", "rb") as f:
        model3_oos_performance = pickle.load(f)
else:
    model3_oos_performance = []

    for look_back_window in look_back_window_list:
        feature_num, oos_error = model3(look_back_window)
        model3_oos_performance.append({"look_back_window": look_back_window, "feature_num": feature_num, "oos_error": oos_error})
    with open("model3_oos_performance.pkl", "wb") as f:
        pickle.dump(model3_oos_performance, f)

#%%
'''
Model 4: principal component regression (PCR) for feature selection, where the coefficients are dependent of stock and independent of time.
Steps:
(1) Construct features by varing the looking-back window for each individual stock;
(2) Calculate the principal components of the features;
(3) Varying the number of principal components to select effective features and perform linear regression;
(4) Evaluate the performance of the model by out-of-sample error and select optimal number of principal features that gives minimum out-of-sample error;
(5) Choose the optimal looking-back window by comparing the out-of-sample error.
'''

def model4(look_back_window):
    print(look_back_window)
    feature_num_all = []
    oos_error_all = []

    for stock_idx in tqdm.tqdm(range(len(id))):
        # train sample
        X = []; Y = []
        for t_idx in range(look_back_window-1, (train_idx[-1])+1, 1):
            ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
            if ar is not None:
                X.append(ar); Y.append(ar2)
        if len(X) == 0:
            continue
        X = np.concatenate(X, axis=0); Y = np.array(Y)
        if X.shape[0] <= X.shape[1]:
            continue

        # test sample
        test_X = []; test_Y = []
        for t_idx in range(test_idx[0], test_idx[-1], 1):
            ar, ar2 = generate_data(stock_idx, t_idx, look_back_window=look_back_window)
            if ar is not None:
                test_X.append(ar); test_Y.append(ar2)
        if len(test_X) == 0:
            continue
        test_X = np.concatenate(test_X, axis=0); test_Y = np.array(test_Y)
        if test_X.shape[0] <= test_X.shape[1]:
            continue

        oos_error = []
        for selected_feature_number in range(1, X.shape[1]):
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            X_reduced = U.dot(np.diag(S))[:, :selected_feature_number].reshape((-1, selected_feature_number))
            linreg = LinearRegression().fit(X_reduced, Y)
            coef = VT[:selected_feature_number, :].reshape((selected_feature_number, -1)).T.dot(linreg.coef_)
            intercept = linreg.intercept_
            predict_Y = test_X.dot(coef) + intercept
            oos_error.append(np.mean(np.abs(predict_Y-test_Y)))
        feature_num_all.append(list(range(1, X.shape[1]))[np.argmin(oos_error)])
        oos_error_all.append(np.min(oos_error))

    return feature_num_all, oos_error_all

look_back_window_list = [1, 2, 3, 4, 5, 10, 30]
if os.path.exists("model4_oos_performance.pkl"):
    with open("model4_oos_performance.pkl", "rb") as f:
        model4_oos_performance = pickle.load(f)
else:
    model4_oos_performance = []
    for look_back_window in look_back_window_list:
        feature_num_all, oos_error_all = model4(look_back_window)
        model4_oos_performance.append({"look_back_window": look_back_window, "feature_num": feature_num_all, "oos_error": oos_error_all})
    with open("model4_oos_performance.pkl", "wb") as f:
        pickle.dump(model4_oos_performance, f)


#%%
'''
By comparing the out-of-sample error of the four linear models, we reach the following conclusions:
(1) the out-of-sample error of all four models consistently decrease as the looking-back window increases -
    --> the long-term dependencies of the features are important to predict the target;
(2) the difference between stock-dependent coefficients and stock-independent coefficients is not significant within same model
    --> we can train the model with data across all stocks
(3) the out-of-sample error of the model 1 and model 3 are almost identical
    --> the principal component regression (PCR) is similar to the Lasso regression for linear models;
(4) the out-of-sample error of the model 4 achives the minimum out-of-sample error
    --> it is desirable to train the model with stock-dependent coefficients. However, it requires a large amount of data for each stock, which may not work well.
'''

plt.figure()
plt.plot(look_back_window_list, [np.min(j["oos_error"]) for j in model1_oos_performance], marker="o", label="model 1=Lasso\n"+ r"($\beta$ universal for all stock)")
plt.plot(look_back_window_list, [np.mean(j["oos_error"]) for j in model2_oos_performance], marker="o", label="model 2=Lasso\n"+r"($\beta$ for each individual stock)")
plt.plot(look_back_window_list, [np.min(j["oos_error"]) for j in model3_oos_performance], marker="o", label="model 3=PCR\n"+r"($\beta$ universal for all stock)")
plt.plot(look_back_window_list, [np.mean(j["oos_error"]) for j in model4_oos_performance], marker="o", label="model=PCR\n"+r"($\beta$ for each individual stock)")
plt.ylabel("out-of-sample error\n"+ r"($l_1$ norm)"); plt.xlabel("look back window")
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("out-of-sample error of linear models.png")

#%%
'''
Best linear model with universal coefficients
Conclusions:
The best linear model with universal coefficients is to 
    apply linear regression to the principal component of the feature matrix whose columns are
    7 feature + daily return + fraction of capitalization for 60 days look-back-window
'''
plt.plot(model3_oos_performance[-1]["feature_num"], model3_oos_performance[-1]["oos_error"])
plt.ylabel("out-of-sample error\n"+ r"($l_1$ norm)"); plt.xlabel("number of principal components")
plt.title("model 3: PCR with universal coefficients")

#%%
'''
Best linear model with stock-dependent coefficients
Conclusions:
The best linear model with stock-dependent coefficents is to 
    apply linear regression to the principal component of the stock-dependent feature matrix whose columns are
    7 feature + daily return + fraction of capitalization for 60 days look-back-window
and select the optimal number of principal components by minimizing the out-of-sample error
'''
plt.scatter(model4_oos_performance[-2]["feature_num"], model4_oos_performance[-2]["oos_error"])
plt.xlabel("number of principal components"); plt.ylabel("out-of-sample error\n"+ r"($l_1$ norm)")
plt.xscale("log")
plt.title("model 4: PCR with stock-dependent coefficients")

# %%



# %%
