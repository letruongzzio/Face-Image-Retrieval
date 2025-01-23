import torch.nn as nn

class RandomDropout:
    """
    A class used to apply random dropout to an image tensor.
    Dropout is a regularization technique where randomly selected neurons are ignored during training. 
    They are "dropped-out" randomly. This prevents overfitting and provides a way to combine many different 
    neural network architectures efficiently.
    Attributes
    ----------
    p : float
        Probability of an element to be zeroed. Default: 0.5
    dropout : nn.Dropout
        An instance of PyTorch's Dropout module initialized with the given probability.
    Methods
    -------
    __call__(img)
        Applies dropout to the input image tensor.
    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from dropout import RandomDropout
    >>> img = torch.randn(1, 3, 224, 224)  # Example image tensor
    >>> random_dropout = RandomDropout(p=0.3)
    >>> output = random_dropout(img)
    >>> print(output.shape)
    torch.Size([1, 3, 224, 224])
    """
    def __init__(self, p=0.5):
        self.p = p
        self.dropout = nn.Dropout(p=p)

    def __call__(self, img):
        return self.dropout(img)