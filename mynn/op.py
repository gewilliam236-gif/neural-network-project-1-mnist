from abc import abstractmethod
import numpy as np

def xavier_uniform(size):
    if len(size) < 2:
        limit = 1.0
    else:
        fan_in = size[0] if len(size) == 2 else np.prod(size[1:])
        fan_out = size[1] if len(size) == 2 else size[0] * np.prod(size[2:])
        limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=size)

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=xavier_uniform, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "forward must be called before backward."
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=xavier_uniform, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, tuple):
            assert kernel_size[0] == kernel_size[1], "Only square kernels are supported."
            kernel_size = kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((1, out_channels, 1, 1))
        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.input_padded = None
        self.output_shape = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4, "conv2D expects input shaped [batch, channels, H, W]."
        batch, channels, height, width = X.shape
        assert channels == self.in_channels
        k = self.kernel_size
        out_h = (height + 2 * self.padding - k) // self.stride + 1
        out_w = (width + 2 * self.padding - k) // self.stride + 1
        assert out_h > 0 and out_w > 0, "Kernel/stride/padding produce invalid output size."

        self.input = X
        if self.padding > 0:
            self.input_padded = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.input_padded = X

        output = np.zeros((batch, self.out_channels, out_h, out_w))
        for i in range(out_h):
            h_start = i * self.stride
            h_end = h_start + k
            for j in range(out_w):
                w_start = j * self.stride
                w_end = w_start + k
                window = self.input_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.tensordot(
                    window, self.W, axes=([1, 2, 3], [1, 2, 3])
                )
        output += self.b
        self.output_shape = output.shape
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None and self.input_padded is not None
        batch, _, out_h, out_w = grads.shape
        k = self.kernel_size

        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True)
        dX_padded = np.zeros_like(self.input_padded)

        for i in range(out_h):
            h_start = i * self.stride
            h_end = h_start + k
            for j in range(out_w):
                w_start = j * self.stride
                w_end = w_start + k
                window = self.input_padded[:, :, h_start:h_end, w_start:w_end]
                grad_ij = grads[:, :, i, j]
                dW += np.tensordot(grad_ij, window, axes=([0], [0]))
                dX_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(
                    grad_ij, self.W, axes=([1], [0])
                )

        self.grads['W'] = dW
        self.grads['b'] = db

        if self.padding > 0:
            return dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)

class Dropout(Layer):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None
        self.training = True
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if not self.training or self.p == 0:
            self.mask = np.ones_like(X)
            return X
        self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
        return X * self.mask

    def backward(self, grads):
        return grads * self.mask

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        assert predicts.shape[0] == labels.shape[0]
        self.predicts = predicts
        self.labels = labels.astype(np.int64)
        batch_size = predicts.shape[0]

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        eps = 1e-12
        correct_probs = self.probs[np.arange(batch_size), self.labels]
        return -np.mean(np.log(correct_probs + eps))
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
