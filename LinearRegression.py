import numpy as np

class LinearRegression():
    def __init__(self, 
                 input_dim: int, 
                 output_dim:int=1, 
                 add_bias:bool=True,
                 lmbd:float=0) -> None:
        """initialize LinearRegression Class

        Args:
            input_dim (int): dim of feature
            output_dim (int): dim of label, the default value is 1
            bias (bool, optional): whether to add bias term. Defaults to True.
            lmbd (float, optional): the coefficent of regularization tern in loss function. \\
                Default is 0. When be assigned not to be zero, the model will be ridge regression.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trained = False
        self.add_bias = add_bias
        if self.add_bias:
            self.weight = np.zeros((input_dim+1, output_dim))
        else:
            self.weight = np.zeros((input_dim, output_dim))
        self.lmbd = lmbd
    
    def fit(self, x:np.array, y:np.array) -> None:
        """fit the model on the training data

        Args:
            x (np.array): feature matrix (m, n)
            y (np.array): label matrix (n, k)
        """
        # reshape y 
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # add bias term to feature matrix
        if self.add_bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))

        # SVD to reduce computation
        U, s, Vh = np.linalg.svd(x, full_matrices=False)
        V = Vh.T
        # get weight
        self.weight = Vh.T @ np.diag([i/(i**2 + self.lmbd) for i in s]) @ U.T @ y
        self.trained = True

    def predict(self, x:np.array) -> np.array:
        """predict the lable of new samples

        Args:
            x (np.array): feature matrix of new samples

        Returns:
            np.array: predicted labels
        """
        if self.trained:
            return x @ self.weight
        else:
            print("The model have not been trained, the output results have no value")
            return x @ self.weight

class LassoRegression():
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 add_bias:bool=True,
                 lmbd:float=1,
                 stop_ctn:float=1e-3,
                 lr:float=1e-3,
                 max_iter:int=1e4) -> None:
        """Initialize Lasso Regression class

        Args:
            input_dim (int): dim of feature
            output_dim (int): dim of label, the default value is 1
            bias (bool, optional): whether to add bias term. Defaults to True.
            lmbd (float, optional): the coefficent of regularization tern in loss function. \\
                Default is 1. When be assigned not to be zero, the model will be ridge regression.
            stop_ctn (float, optional): Default is 1e-3. Be used to determine whether to stop training or not.
            lr (float, optinal): Default is 1e-3. Learning rate.
            max_iter (int, optional): Default is 1e4. # of max iteration times
        """
        self.input_dim = input_dim
        self.outpot_dim = output_dim
        self.lmbd = lmbd
        self.trained = False
        self.add_bias = add_bias
        if self.add_bias:
            self.weight = np.ones(shape=(input_dim+1, output_dim))
        else:
            self.weight = np.ones(shape=(input_dim, output_dim))
        self.stop_ctn = stop_ctn
        self.lr = lr
        self.max_iter = max_iter

    def loss(self, x:np.array, y:np.array) -> float:
        """calculate the loss of the model.

        Args:
            x (np.array): feautre matrix
            y (np.array): output label

        Returns:
            float: loss 
        """
        return np.sum((x @ self.weight - y)**2) + self.lmbd * sum(np.abs(self.weight))
        
        
    
    def fit(self, x:np.array, y:np.array) -> None:
        """fit the model on the training data

        Args:
            x (np.array): input feature matrix
            y (np.array): output label
        """
        # reshape y 
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        sample_num = x.shape[0]
        # add bias term to feature matrix
        if self.add_bias:
            x = np.hstack((x, np.ones((sample_num, 1))))

        pre_loss = np.inf
        cur_loss = self.loss(x, y)
        loss_diff = pre_loss - cur_loss

        U, s, Vh = np.linalg.svd(x, full_matrices=False)

        iter = 0

        while abs(loss_diff) > self.stop_ctn:
            weight_abs = np.abs(self.weight)
            weight_abs[weight_abs == 0] = 1
            grad = 2 * Vh.T @ np.diag([i**2 for i in s]) @ Vh @ self.weight - 2 * x.T @ y + self.lmbd * self.weight / weight_abs
            self.weight = self.weight - self.lr * grad / sample_num
            pre_loss = cur_loss
            cur_loss = self.loss(x, y)
            loss_diff = pre_loss - cur_loss
            iter =+ 1
            if iter >= self.max_iter:
                break
        self.trained = True
        
    def predict(self, x:np.array) -> np.array:
        """predict the label of new samples

        Args:
            x (np.array): feature matrix of new samples

        Returns:
            np.array: predicted labels
        """
        if self.add_bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))

        if self.trained:
            return x @ self.weight
        else:
            print("The model have not been trained, the output results have no value")
            return x @ self.weight


