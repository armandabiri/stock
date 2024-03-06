import numpy as np
class UnscentedKalmanFilter:
    def __init__(self, f, h, x, Pxx, Q=None, R=None, dt=None, alpha=1e-3, beta=2, kappa=0):
        self.f = f  # System model function handle (x, u, dt)
        self.h = h  # Measurement model function handle (x)
        self.x = x  # Current state estimate
        self.Pxx = Pxx  # Current state covariance
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.dt = dt  # Time step
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def update(self, dt, z=None):
        n = len(self.x)  # Number of state variables

        # Unscented transformation parameters
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        gamma = n + lambda_

        # Calculate square root of gamma * Pxx
        flag=False
        iter=0
        while (flag == False) and (iter < 100):
            try:
                sP = np.linalg.cholesky(gamma * self.Pxx)
                iter += 1
                flag = True
            except np.linalg.LinAlgError:
                flag = False
                self.Pxx = self.nearestcorr(self.Pxx, method='eigen2')



        # Compute the sigma points
        X = np.hstack((self.x[:, np.newaxis] - sP, self.x[:, np.newaxis], self.x[:, np.newaxis] + sP))

        Wm = np.ones(2 * n + 1) / (2 * gamma)
        Wm[n] = lambda_ / gamma

        Wc = Wm.copy()
        Wc[n] = Wc[n] + (1 - self.alpha**2 + self.beta)

        # Predict
        zpp=self.h(dt,X[:, 0])
         # Check if z is a single float
        if isinstance(zpp, (int, float)):
            m = 1
        else:
            m = len(zpp)

        Z = np.zeros((m, 2 * n + 1))
        for i in range(2 * n + 1):
            X[:, i] = self.f(dt,X[:, i])
            Z[:, i] = self.h(dt,X[:, i])

        x_mean = X @ Wm
        z_mean = Z @ Wm

        X_error = X - x_mean[:, np.newaxis]
        Pxx = X_error @ np.diag(Wm) @ X_error.T + self.Q

        Z_error = Z - z_mean[:, np.newaxis]
        Pzz = Z_error @ np.diag(Wm) @ Z_error.T + self.R

        Pxz = X_error @ np.diag(Wm) @ Z_error.T

        if z is None:
            z=z_mean

        K = Pxz @ np.linalg.inv(Pzz)  # Kalman gain

        self.x = x_mean + K @ (z - z_mean)  # Update
        S1 = K @ Pzz.T @ K.T
        S2 = K @ Pxz.T
        S = (S1 + S2) / 2

        self.Pxx = Pxx - S

        if np.linalg.norm(self.Pxx) > 1e4:
            U, S, V = np.linalg.svd(self.Pxx)
            d = np.diag(S)
            d = d / np.linalg.norm(d)
            self.Pxx = U @ np.diag(d) @ np.linalg.inv(V)

        return self.x, self.Pxx
    def nearestcorr(A, **kwargs):
        method = kwargs.get('method', 'eigen')

        if method == 'eigen':
            V, D = np.linalg.eig(A)
            lambda_ = np.diag(D)
            lambda_[lambda_ < 0] = 1e-6
            Ac = np.dot(V, np.dot(np.diag(lambda_), np.linalg.pinv(V)))
            Ac = np.real(Ac)
        elif method == 'eigen2':
            V, D = np.linalg.eig(A)
            lambda_ = np.diag(D)
            lambda_[lambda_ < 0] = 1e-6
            S = np.zeros_like(A)
            for i in range(A.shape[0]):
                for j in range(A.shape[0]):
                    S[i, i] += V[j, i] ** 2 * lambda_[j]
                S[i, i] = np.sqrt(1 / S[i, i])
            Ac = np.dot(np.dot(S, V), np.diag(np.sqrt(lambda_)))
        else:
            raise ValueError("Invalid method specified")

        return Ac
