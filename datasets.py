import matplotlib.pyplot as plt
import sklearn.datasets as skds
import numpy as np

def load_flower_dataset(num_samples=500, petals=4, petal_length=4, noise=0.2, angle=30):
    np.random.seed(1)
    m = num_samples # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*np.pi,(j+1)*np.pi,N) + np.random.randn(N)*noise # theta, classes mixing imperfection
        r = petal_length*np.sin(petals*t) + np.random.randn(N)*noise # radius, petal shape imperfection
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    # rotating data points
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    X = np.dot(X, rotation_matrix)
    # previous shape is (num_samples, dim), we need to transpose it to (dim, num_samples
    X = X.T
    # previous shape is (num_samples,), we need to reshape it to (1, num_samples)
    Y = Y.T

    return X, Y


def uneven_distr_flower_dataset(num_samples=500, petals=4, petal_length=4, noise=0.1, angle=30):
    """
    Generate synthetic 2D flower-shaped data for binary classification.


    Args:
        num_samples (int): Number of data samples to generate.
        noise (float): Amount of noise to add to the data.

    Returns:
        X (numpy.ndarray): Input features of shape (2, num_samples).
        Y (numpy.ndarray): Target labels of shape (1, num_samples,).
    """
    np.random.seed(1)
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    theta = np.linspace(0,2*np.pi,num_samples) + np.random.randn(num_samples)*noise
    r = petal_length * np.sin(petals * theta) + np.random.randn(num_samples)*noise
    X = np.zeros((num_samples, 2))
    X[:, 0] = r * np.cos(theta)
    X[:, 1] = r * np.sin(theta)
    Y = np.where(theta < np.pi, 0, 1)
    
    # rotating data points
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    X = np.dot(X, rotation_matrix)
    # previous shape is (num_samples, dim), we need to transpose it to (dim, num_samples
    X = X.T
    # previous shape is (num_samples,), we need to reshape it to (1, num_samples)
    Y = Y.T.reshape(1,-1)
    return X, Y


def load_spiral_dataset(N=1000, K=4, noise=0.1):
    N=int(N/K) # number of points per class
    D = 2
    X = np.zeros((N * K, D)) # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1.0, N) # radius
        cir_div = 2*np.pi/K
        t = np.linspace(j * cir_div, (j + 1) * cir_div, N) + np.random.randn(N) * noise # theta
        X[ix] = np.c_[r * np.sin(t), r*np.cos(t)]
        y[ix] = j
    
    return X.T, y.reshape(1,-1)


def load_two_spirals_dataset(n_points=1000, noise=.5, rev=2.0):
    np.random.seed(3)
    nn = np.random.uniform(0.01, 1.0, (n_points,1))
    n = np.sqrt(nn) * rev * (2*np.pi)
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    x= np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y))))
    y= np.hstack((np.zeros(n_points),np.ones(n_points)))
    return x.T,y.T


def load_circles_dataset(n_samples=500, factor=.5, noise=.1, random_state=42):
    x, y = skds.make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=random_state)
    return x.T, y.reshape(1,-1)


def load_moons_dataset(n_samples=500, noise=.1, random_state=42):
    x, y = skds.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return x.T, y.reshape(1,-1)


def load_blobs_dataset(n_samples=500, n_features=2, centers=2, cluster_std=1.0, random_state=42):
    x, y = skds.make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return x.T, y.reshape(1,-1)


def load_gaussian_quantiles_dataset(n_samples=500, n_features=2, n_classes=3, mean=None, cov=1.0, random_state=42):
    x, y = skds.make_gaussian_quantiles(n_samples=n_samples, n_features=n_features, n_classes=n_classes, mean=mean, cov=cov, random_state=random_state)
    return x.T, y.reshape(1,-1)


def load_no_structure_dataset(N=500):
    return np.random.rand(2, N), np.round(np.random.rand(1, N)).astype(int)

