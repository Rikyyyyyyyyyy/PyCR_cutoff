import numpy as np
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression


def selrpy(X, y, ncomponents):
    X = np.array(X)
    pls = PLSRegression(n_components=ncomponents)
    pls.fit_transform(scale(X,axis=0,with_mean=True,with_std=True), y)
    bpls = pls.coef_
    xw = pls.x_weights_
    xw = (xw / np.linalg.norm(xw, axis=0))

    ttp = X @ (bpls / np.linalg.norm(bpls, axis=0))
    ptp = X.T @ ttp @ np.linalg.pinv(ttp.T @ ttp)

    mdlErr = X - ttp @ ptp.T

    selr = []

    for ii in range(np.size(X, 1)):
        ssExp = np.linalg.norm(ttp @ ptp[ii, :].T) ** 2
        ssRes = np.linalg.norm(mdlErr[:, ii]) ** 2
        selr = list(selr)
        selr.append(np.divide(ssExp, ssRes))
        selr = np.array(selr)
        selr = np.nan_to_num(selr, nan=(10 ** -12))


    return selr



