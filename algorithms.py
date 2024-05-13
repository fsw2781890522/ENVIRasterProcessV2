# import torch
# from numba import jit
# import cupy as cp
# from torch import Tensor

import numpy as np
from tqdm import tqdm

"""-------------------------------------denoise algorithms------------------------------------"""


def pca_transform(data):
    """
    ENVI official docs:

        PCA is a linear transformation that reorganizes the variance in a multiband image into a new set of image bands.
        These PC bands are uncorrelated linear combinations of the input bands.
        A PC transform finds a new set of orthogonal axes with their origin at the data mean,
        and it rotates them so the data variance is maximized.
        A more detailed discussion of PCA is available in most remote sensing literature.

        ENVI performs the following steps to perform PCA:

        1.Compute the input image covariance or correlation matrix, depending on user preference.
        2.Compute the eigenvectors of the covariance or correlation matrix.
            See View PC Statistics below for more information on interpreting these statistics.
        3.Subtract the band mean from the input image data.
            This correction produces an origin shift in the output PC image such that its mean spectrum is 0 in every band.
        4.Project the mean-corrected image data onto the transpose of the eigenvector matrix,
            using the same approach as Richards (1999) but using the following equation:

            y = G # (x-mean)

            Where:

            y = Transformed (or rotated) data

            G = Transformation matrix

            x = Input data

            # Denotes matrix multiplication

        An inverse PC rotation is computed by projecting the PC-rotated image data onto the inverse of the PCA transformation matrix.
    """
    # 将数据重新排列为 (nPixels, nBands) 的形状
    nRows, nColumns, nBands = data.shape
    reshaped_data = np.reshape(data, (nRows * nColumns, nBands))

    # 计算均值
    print('Calculating means...')
    mean = np.mean(reshaped_data, axis=0)

    # 中心化数据
    print('Centralizing data...')
    centered_data = reshaped_data - mean

    # 释放不再需要的变量
    del reshaped_data

    # 计算协方差矩阵
    print('Calculating covariance matrix...')
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # 计算特征值和特征向量
    print('Calculating eigenvalues and eigenvectors...')
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 释放不再需要的变量
    del covariance_matrix

    # 对特征向量排序
    print('Sorting eigenvectors...')
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    class Sta:
        def __init__(self, mean, eigenvectors):
            Sta.mean = mean
            Sta.eigenvectors = eigenvectors

    sta = Sta(mean, eigenvectors)

    # 计算正变换后的数据
    print('Transforming...')
    transformed_data = np.dot(centered_data, eigenvectors)
    # 将transformed_data重新排列为与原始数据形状一致的三维数组
    transformed_data = np.reshape(transformed_data, (nRows, nColumns, nBands))

    return transformed_data, sta


def inverse_pca_transform(transformed_data, sta):
    # 使用逆变换重构数据
    reconstructed_data = np.dot(transformed_data, sta.eigenvectors.T) + sta.mean

    return reconstructed_data


def cal_stats(window):
    """
    :param window: a sliding 2D-Array, same nRows as image, assigned nColumns by user(default: same nColumns as image)
        extracted from a single band of an image
    :return: an object with attributes miu_ref, sigma_ref, miu, sigma
    miu_ref: a single number. the mean value of total window
    sigma_ref: a single number. the std value of total window
    miu: an array which records mean values of each column in window
    sigma: an array which records std values of each column in window
    """
    miu_sum = 0.0
    sigma_sum = 0.0
    miu = []
    sigma = []

    for j in range(window.shape[1]):
        column = window[:, j]

        ht, locations = np.histogram(column, bins=200)
        ht_acc = np.cumsum(ht) / np.prod(column.shape)
        """
        ht_acc: accumulative frequency of histogram
        note: ht, locations = np.histogram(column, bins=200, density=true) easily cause error
        """
        w1 = np.where(ht_acc >= 0.99)
        w2 = np.where(ht_acc >= 0.01)
        max_val = locations[w1[0][0]]
        min_val = locations[w2[0][0]]

        w = np.where((column >= min_val) & (column <= max_val))
        temp = column[w]
        miu.append(np.nanmean(temp))
        sigma.append(np.nanstd(temp))
        miu_sum += np.nanmean(temp)
        sigma_sum += np.nanstd(temp)

    miu_ref = miu_sum / float(window.shape[1])
    sigma_ref = sigma_sum / float(window.shape[1])

    class Stats:
        def __init__(self, miu_ref, sigma_ref, miu, sigma):
            self.miu_ref = miu_ref
            self.sigma_ref = sigma_ref
            self.miu = miu
            self.sigma = sigma

    return Stats(miu_ref, sigma_ref, miu, sigma)


def match(arr, width):
    """
    To moment match a single band in an image.
    :param arr: a 2D-array. a single band data
    :param width: the width a sliding window in which moment match is applied
    :return: a moment matched 2D-array as a processed single band
    """
    nl, ns = arr.shape
    result_arr = arr

    for i in range(ns - width + 1):
        window = arr[:, i: i + width]
        stats = cal_stats(window)
        miu_ref = stats.miu_ref
        sigma_ref = stats.sigma_ref
        miu_vals = stats.miu
        sigma_vals = stats.sigma

        for j in range(window.shape[1]):
            # check if sigma_vals[j] is zero to avoid divide by zero error
            if sigma_vals[j] != 0:
                gain = sigma_ref / sigma_vals[j]
                offset = miu_ref - (gain * miu_vals[j])
                # prevent from extreme coefficient
                if gain <= 5:
                    window[:, j] = gain * window[:, j] + offset

        result_arr[:, i: i + width] = window

    return result_arr


# def denoise_gpu(data, width=None):
#     """
#
#     :param width:
#     :param data: numpy ndarray
#     :return:
#     """
#     # 将数据重新排列为 (cpixels, nBands) 的形状
#     nl, ns, nb = data.shape
#     reshaped_data = cp.reshape(cp.asarray(data), (nl * ns, nb))
#
#     # 计算均值
#     print('Calculating means...')
#     mean = cp.mean(reshaped_data, axis=0)
#
#     # 中心化数据
#     print('Centralizing data...')
#     centered_data = reshaped_data - mean
#
#     # 释放不再需要的变量
#     reshaped_data = None
#
#     # 计算协方差矩阵
#     print('Calculating covariance matrix...')
#     covariance_matrix = cp.cov(centered_data, rowvar=False)
#
#     # 计算特征值和特征向量
#     print('Calculating eigenvalues and eigenvectors...')
#     eigenvalues, eigenvectors = cp.linalg.eigh(covariance_matrix)
#
#     # 释放不再需要的变量
#     covariance_matrix = None
#
#     # 对特征向量排序
#     print('Sorting eigenvectors...')
#     sorted_indices = cp.argsort(eigenvalues)[::-1]
#     eigenvectors = eigenvectors[:, sorted_indices]
#
#     # 计算正变换后的数据
#     print('Forward PCA transforming...')
#     pca_data = cp.reshape(cp.dot(centered_data, eigenvectors), (nl, ns, nb))
#
#     print('Moment matching...')
#     if width is None:
#         width = int(ns)
#
#     matched_data = cp.zeros((nl, ns, nb), dtype=cp.float32)
#
#     for band in range(nb):
#
#         band_data = cp.reshape(pca_data[:, :, band], (nl, ns))
#
#         for i in range(ns - width + 1):
#             window = band_data[:, i: i + width]
#             miu_sum = 0.0
#             sigma_sum = 0.0
#             miu = []
#             sigma = []
#
#             for j in range(window.shape[1]):
#                 column = window[:, j]
#
#                 ht, locations = cp.histogram(column, bins=200)
#                 ht_acc = cp.cumsum(ht) / cp.prod(column.shape)
#                 """
#                 ht_acc: accumulative frequency of histogram
#                 note: ht, locations = cp.histogram(column, bins=200, density=true) easily cause error
#                 """
#                 w1 = cp.where(ht_acc >= 0.99)
#                 w2 = cp.where(ht_acc >= 0.01)
#                 max_val = locations[w1[0][0]]
#                 min_val = locations[w2[0][0]]
#
#                 w = cp.where(cp.logical_and((column >= min_val), (column <= max_val)))
#                 temp = column[w]
#                 miu.append(cp.nanmean(temp))
#                 sigma.append(cp.nanstd(temp))
#                 miu_sum += cp.nanmean(temp)
#                 sigma_sum += cp.nanstd(temp)
#
#             miu_ref = miu_sum / float(window.shape[1])
#             sigma_ref = sigma_sum / float(window.shape[1])
#
#             for j in range(window.shape[1]):
#                 # check if sigma_vals[j] is zero to avoid divide by zero error
#                 if sigma[j] != 0:
#                     gain = sigma_ref / sigma[j]
#                     offset = miu_ref - (gain * miu[j])
#                     # prevent from extreme coefficient
#                     if gain <= 5:
#                         window[:, j] = gain * window[:, j] + offset
#
#             matched_data[:, i: i + width, band] = window
#
#     pca_data = None
#
#     inverse_transformed_data = cp.dot(matched_data, eigenvectors.T) + mean
#
#     return cp.reshape(inverse_transformed_data, (nl, ns, nb))


"""-------------------------------------SPy algorithms------------------------------------"""

# class PrincipalComponents:
#     '''
#     An object for storing a data set's principal components.  The
#     object has the following members:
#
#         `eigenvalues`:
#
#             A length B array of eigenvalues sorted in descending order
#
#         `eigenvectors`:
#
#             A `BxB` array of normalized eigenvectors (in columns)
#
#         `stats` (:class:`GaussianStats`):
#
#             A statistics object containing `mean`, `cov`, and `nsamples`.
#
#         `transform`:
#
#             A callable function to transform data to the space of the
#             principal components.
#
#         `reduce`:
#
#             A method to return a reduced set of principal components based
#             on either a fixed number of components or a fraction of total
#             variance.
#
#         `denoise`:
#
#             A callable function to denoise data using a reduced set of
#             principal components.
#
#         `get_denoising_transform`:
#
#             A callable function that returns a function for denoising data.
#     '''
#
#     def __init__(self, vals, vecs, stats):
#         self.eigenvalues = vals
#         self.eigenvectors = vecs
#         self.stats = stats
#         self.transform = LinearTransform(self.eigenvectors.T, pre=-self.mean)
#
#     @property
#     def mean(self):
#         return self.stats.mean
#
#     @property
#     def cov(self):
#         return self.stats.cov
#
#     def reduce(self, N=0, **kwargs):
#         '''Reduces the number of principal components.
#
#         Keyword Arguments (one of the following must be specified):
#
#             `num` (integer):
#
#                 Number of eigenvalues/eigenvectors to retain.  The top `num`
#                 eigenvalues will be retained.
#
#             `eigs` (list):
#
#                 A list of indices of eigenvalues/eigenvectors to be retained.
#
#             `fraction` (float):
#
#                 The fraction of total image variance to retain.  Eigenvalues
#                 will be retained (starting from greatest to smallest) until
#                 `fraction` of total image variance is retained.
#         '''
#         status = spy._status
#
#         num = kwargs.get('num', None)
#         eigs = kwargs.get('eigs', None)
#         fraction = kwargs.get('fraction', None)
#         if num is not None:
#             return PrincipalComponents(self.eigenvalues[:num],
#                                        self.eigenvectors[:, :num],
#                                        self.stats)
#         elif eigs is not None:
#             vals = self.eigenvalues[eigs]
#             vecs = self.eigenvectors[:, eigs]
#             return PrincipalComponents(vals, vecs, self.stats)
#         elif fraction is not None:
#             if not 0 < fraction <= 1:
#                 raise Exception('fraction must be in range (0,1].')
#             N = len(self.eigenvalues)
#             cumsum = np.cumsum(self.eigenvalues)
#             sum = cumsum[-1]
#             # Count how many values to retain.
#             for i in range(N):
#                 if (cumsum[i] / sum) >= fraction:
#                     break
#             if i == (N - 1):
#                 # No reduction
#                 status.write('No reduction in eigenvectors achieved.')
#                 return self
#
#             vals = self.eigenvalues[:i + 1]
#             vecs = self.eigenvectors[:, :i + 1]
#             return PrincipalComponents(vals, vecs, self.stats)
#         else:
#             raise Exception('Must specify one of the following keywords:'
#                             '`num`, `eigs`, `fraction`.')
#
#     def denoise(self, X, **kwargs):
#         '''Returns a de-noised version of `X`.
#
#         Arguments:
#
#             `X` (np.ndarray):
#
#                 Data to be de-noised. Can be a single pixel or an image.
#
#         Keyword Arguments (one of the following must be specified):
#
#             `num` (integer):
#
#                 Number of eigenvalues/eigenvectors to use.  The top `num`
#                 eigenvalues will be used.
#
#             `eigs` (list):
#
#                 A list of indices of eigenvalues/eigenvectors to be used.
#
#             `fraction` (float):
#
#                 The fraction of total image variance to retain.  Eigenvalues
#                 will be included (starting from greatest to smallest) until
#                 `fraction` of total image variance is retained.
#
#         Returns denoised image data with same shape as `X`.
#
#         Note that calling this method is equivalent to calling the
#         `get_denoising_transform` method with same keyword and applying the
#         returned transform to `X`. If you only intend to denoise data with the
#         same parameters multiple times, then it is more efficient to get the
#         denoising transform and reuse it, rather than calling this method
#         multilple times.
#         '''
#         f = self.get_denoising_transform(**kwargs)
#         return f(X)
#
#     def get_denoising_transform(self, **kwargs):
#         '''Returns a function for denoising image data.
#
#         Keyword Arguments (one of the following must be specified):
#
#             `num` (integer):
#
#                 Number of eigenvalues/eigenvectors to use.  The top `num`
#                 eigenvalues will be used.
#
#             `eigs` (list):
#
#                 A list of indices of eigenvalues/eigenvectors to be used.
#
#             `fraction` (float):
#
#                 The fraction of total image variance to retain.  Eigenvalues
#                 will be included (starting from greatest to smallest) until
#                 `fraction` of total image variance is retained.
#
#         Returns a callable :class:`~spectral.algorithms.transforms.LinearTransform`
#         object for denoising image data.
#         '''
#         V = self.reduce(self, **kwargs).eigenvectors
#         f = LinearTransform(V.dot(V.T), pre=-self.mean,
#                             post=self.mean)
#         return f
#
#
# class LinearTransform:
#     '''A callable linear transform object.
#
#     In addition to the __call__ method, which applies the transform to given,
#     data, a LinearTransform object also has the following members:
#
#         `dim_in` (int):
#
#             The expected length of input vectors. This will be `None` if the
#             input dimension is unknown (e.g., if the transform is a scalar).
#
#         `dim_out` (int):
#
#             The length of output vectors (after linear transformation). This
#             will be `None` if the input dimension is unknown (e.g., if
#             the transform is a scalar).
#
#         `dtype` (numpy dtype):
#
#             The numpy dtype for the output ndarray data.
#     '''
#
#     def __init__(self, A, **kwargs):
#         '''Arguments:
#
#             `A` (:class:`~numpy.ndarrray`):
#
#                 An (J,K) array to be applied to length-K targets.
#
#         Keyword Argments:
#
#             `pre` (scalar or length-K sequence):
#
#                 Additive offset to be applied prior to linear transformation.
#
#             `post` (scalar or length-J sequence):
#
#                 An additive offset to be applied after linear transformation.
#
#             `dtype` (numpy dtype):
#
#                 Explicit type for transformed data.
#         '''
#
#         self._pre = kwargs.get('pre', None)
#         self._post = kwargs.get('post', None)
#         A = np.array(A, copy=True)
#         if A.ndim == 0:
#             # Do not know input/ouput dimensions
#             self._A = A
#             (self.dim_out, self.dim_in) = (None, None)
#         else:
#             if len(A.shape) == 1:
#                 self._A = A.reshape(((1,) + A.shape))
#             else:
#                 self._A = A
#             (self.dim_out, self.dim_in) = self._A.shape
#         self.dtype = kwargs.get('dtype', self._A.dtype)
#
#     def __call__(self, X):
#         '''Applies the linear transformation to the given data.
#
#         Arguments:
#
#             `X` (:class:`~numpy.ndarray` or object with `transform` method):
#
#                 If `X` is an ndarray, it is either an (M,N,K) array containing
#                 M*N length-K vectors to be transformed or it is an (R,K) array
#                 of length-K vectors to be transformed. If `X` is an object with
#                 a method named `transform` the result of passing the
#                 `LinearTransform` object to the `transform` method will be
#                 returned.
#
#         Returns an (M,N,J) or (R,J) array, depending on shape of `X`, where J
#         is the length of the first dimension of the array `A` passed to
#         __init__.
#         '''
#         if not isinstance(X, np.ndarray):
#             if hasattr(X, 'transform') and isinstance(X.transform, Callable):
#                 return X.transform(self)
#             else:
#                 raise TypeError('Unable to apply transform to object.')
#
#         shape = X.shape
#         if len(shape) == 3:
#             X = X.reshape((-1, shape[-1]))
#             if self._pre is not None:
#                 X = X + self._pre
#             Y = np.dot(self._A, X.T).T
#             if self._post is not None:
#                 Y += self._post
#             return Y.reshape((shape[:2] + (-1,))).squeeze().astype(self.dtype)
#         else:
#             if self._pre is not None:
#                 X = X + self._pre
#             Y = np.dot(self._A, X.T).T
#             if self._post is not None:
#                 Y += self._post
#             return Y.astype(self.dtype)
#
#     def chain(self, transform):
#         '''Chains together two linear transforms.
#         If the transform `f1` is given by
#
#         .. math::
#
#             F_1(X) = A_1(X + b_1) + c_1
#
#         and `f2` by
#
#         .. math::
#
#             F_2(X) = A_2(X + b_2) + c_2
#
#         then `f1.chain(f2)` returns a new LinearTransform, `f3`, whose output
#         is given by
#
#         .. math::
#
#             F_3(X) = F_2(F_1(X))
#         '''
#
#         if isinstance(transform, np.ndarray):
#             transform = LinearTransform(transform)
#         if self.dim_in is not None and transform.dim_out is not None \
#                 and self.dim_in != transform.dim_out:
#             raise Exception('Input/Output dimensions of chained transforms'
#                             'do not match.')
#
#         # Internally, the new transform is computed as:
#         # Y = f2._A.dot(f1._A).(X + f1._pre) + f2._A.(f1._post + f2._pre) + f2._post
#         # However, any of the _pre/_post members could be `None` so that needs
#         # to be checked.
#
#         if transform._pre is not None:
#             pre = np.array(transform._pre)
#         else:
#             pre = None
#         post = None
#         if transform._post is not None:
#             post = np.array(transform._post)
#             if self._pre is not None:
#                 post += self._pre
#         elif self._pre is not None:
#             post = np.array(self._pre)
#         if post is not None:
#             post = self._A.dot(post)
#         if self._post:
#             post += self._post
#         if post is not None:
#             post = np.array(post)
#         A = np.dot(self._A, transform._A)
#         return LinearTransform(A, pre=pre, post=post)
#
#
# def calc_stats(image, mask=None, index=None, allow_nan=False):
#     '''Computes Gaussian stats for image data..
#
#     Arguments:
#
#         `image` (ndarrray, :class:`~spectral.Image`, or :class:`spectral.Iterator`):
#
#             If an ndarray, it should have 2 or 3 dimensions and the mean &
#             covariance will be calculated for the last dimension.
#
#         `mask` (ndarray):
#
#             If `mask` is specified, mean & covariance will be calculated for
#             all pixels indicated in the mask array.  If `index` is specified,
#             all pixels in `image` for which `mask == index` will be used;
#             otherwise, all nonzero elements of `mask` will be used.
#
#         `index` (int):
#
#             Specifies which value in `mask` to use to select pixels from
#             `image`. If not specified but `mask` is, then all nonzero elements
#             of `mask` will be used.
#
#         `allow_nan` (bool, default False):
#
#             If True, statistics will be computed even if `np.nan` values are
#             present in the data; otherwise, `~spectral.algorithms.spymath.NaNValueError`
#             is raised.
#
#         If neither `mask` nor `index` are specified, all samples in `vectors`
#         will be used.
#
#     Returns:
#
#         `GaussianStats` object:
#
#             This object will have members `mean`, `cov`, and `nsamples`.
#     '''
#     (mean, cov, N) = mean_cov(image, mask, index)
#     if has_nan(mean) and not allow_nan:
#         raise NaNValueError('NaN values present in data.')
#     return GaussianStats(mean=mean, cov=cov, nsamples=N)
#
#
# def principal_components(image):
#     '''
#     Calculate Principal Component eigenvalues & eigenvectors for an image.
#
#     Usage::
#
#         pc = principal_components(image)
#
#     Arguments:
#
#         `image` (ndarray, :class:`spectral.Image`, :class:`GaussianStats`):
#
#             An `MxNxB` image
#
#     Returns a :class:`~spectral.algorithms.algorithms.PrincipalComponents`
#     object with the following members:
#
#         `eigenvalues`:
#
#             A length B array of eigenvalues
#
#         `eigenvectors`:
#
#             A `BxB` array of normalized eigenvectors
#
#         `stats` (:class:`GaussianStats`):
#
#             A statistics object containing `mean`, `cov`, and `nsamples`.
#
#         `transform`:
#
#             A callable function to transform data to the space of the
#             principal components.
#
#         `reduce`:
#
#             A method to reduce the number of eigenvalues.
#
#         `denoise`:
#
#             A callable function to denoise data using a reduced set of
#             principal components.
#
#         `get_denoising_transform`:
#
#             A callable function that returns a function for denoising data.
#     '''
#     if isinstance(image, GaussianStats):
#         stats = image
#     else:
#         stats = calc_stats(image)
#
#     (L, V) = np.linalg.eig(stats.cov)
#
#     # numpy says eigenvalues may not be sorted so we'll sort them, if needed.
#     if not np.alltrue(np.diff(L) <= 0):
#         ii = list(reversed(np.argsort(L)))
#         L = L[ii]
#         V = V[:, ii]
#
#     return PrincipalComponents(L, V, stats)
