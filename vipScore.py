import numpy as np

from sklearn.cross_decomposition import PLSRegression


def vipy(X, y, ncomponents):
    vipDic = {}
    pls = PLSRegression(n_components=ncomponents)
    (xscrs, yscrs) = pls.fit_transform(X, y)
    xw = pls.x_weights_
    b = np.linalg.pinv(xscrs) @ y
    xw = (xw / np.linalg.norm(xw, axis=0)) ** 2
    sz = np.size(X, 1)
    vipscrs = []

    for vrbl in range(sz):
        nmtr = np.sum((b**2) @ xscrs.T @ xscrs @ xw[vrbl, :])
        dmtr = np.sum((b**2) @ xscrs.T @ xscrs)
        vipscrs.append(np.sqrt((sz * nmtr) / dmtr))
    for i in range(len(X[0])):
        vipDic[i] = vipscrs[i]

    return vipDic