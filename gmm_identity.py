import math
from scipy.stats import multivariate_normal


def get_identity(x, n, weights, means, covars):
    log_sum = [0] * n
    for idx in range(n):
        for i in range(x.shape[0]):
            pb_sum = 0
            for j in range(32):
                p = weights[idx][j] * multivariate_normal.pdf(x[i], mean=means[idx][j], cov=covars[idx][j])
                pb_sum = p + pb_sum
            if pb_sum > 0:
                log_sum[idx] = math.log(pb_sum) + log_sum[idx]
            else:
                log_sum[idx] = -math.inf
                print("WARNING: unexpected negative number: ", pb_sum)
    return log_sum.index(max(log_sum))
