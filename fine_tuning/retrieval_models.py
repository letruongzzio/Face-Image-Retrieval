import torch.nn as nn
from torchvision import models

class RetrievalModel(nn.Module):
    """
    RetrievalModel is a neural network module designed for image retrieval tasks.
    This model leverages a pre-trained backbone architecture (MobileNetV2 or ResNet50) 
    to extract feature representations from input images. It then projects these features 
    into a lower-dimensional embedding space using an embedding layer. The embeddings can 
    be used for tasks such as image similarity search, clustering, or other retrieval-based 
    applications.
        backbone (str, optional): Backbone architecture for feature extraction. Supported 
        options are 'mobilenet_v2' and 'resnet50'. Defaults to 'mobilenet_v2'.
        embedding_dim (int, optional): Dimension of the embedding space. Determines the size 
        of the output embedding vector. Defaults to 128.
    Attributes:
        backbone (nn.Module): The feature extraction backbone, either MobileNetV2 or ResNet50 
        without the final classification layers.
        embedding (nn.Sequential): Sequential container consisting of an adaptive average pooling 
        layer, flattening layer, and two linear layers with ReLU activation in between for embedding generation.
    Methods:
        forward(x): Defines the forward pass of the model, processing input images to produce embedding vectors.
    Example:
        ```python
        # Initialize the model with ResNet50 backbone
        model = RetrievalModel(backbone='resnet50', embedding_dim=256, num_attributes=50)
        model.eval()
        # Create a dummy input tensor with batch size 8 and 3 color channels, 224x224 pixels
        dummy_input = torch.randn(8, 3, 224, 224)
        # Generate embeddings
        embeddings = model(dummy_input)
        print(embeddings.shape)  # Expected output: torch.Size([8, 256])
        ```
    """
    def __init__(self, backbone='mobilenet_v2', embedding_dim=128):
        """
        Initializes the RetrievalModel with the specified backbone architecture and embedding dimension.

            backbone (str, optional): The backbone architecture for feature extraction. Supported options are:
                - 'mobilenet_v2': Utilizes the MobileNetV2 architecture with pretrained weights.
                - 'resnet50': Utilizes the ResNet50 architecture with pretrained weights.
                Defaults to 'mobilenet_v2'.
            embedding_dim (int, optional): The dimensionality of the embedding space. Determines the size of 
            the output embedding vectors.
                Defaults to 128.

        Raises:
            ValueError: If an unsupported backbone architecture is specified.

        Attributes:
            backbone (nn.Module): The feature extraction backbone model, excluding the final classification layers.
            embedding (nn.Sequential): A sequential container comprising:
                - `nn.AdaptiveAvgPool2d(1)`: Pools the spatial dimensions to a 1x1 output.
                - `nn.Flatten()`: Flattens the pooled output to a one-dimensional tensor.
                - `nn.Linear(num_features, embedding_dim)`: Projects the features to the specified embedding dimension.
                - `nn.ReLU()`: Applies the ReLU activation function.
                - `nn.Linear(embedding_dim, embedding_dim)`: Further projects to the embedding dimension for enhanced 
                feature representation.

        Example:
            ```python

            # Initialize the model with ResNet50 backbone and embedding dimension of 256
            model = RetrievalModel(backbone='resnet50', embedding_dim=256)
            model.eval()

            # Create a dummy input tensor with batch size 8 and 3 color channels, 224x224 pixels
            dummy_input = torch.randn(8, 3, 224, 224)

            # Generate embeddings
            embeddings = model(dummy_input)
            print(embeddings.shape)  # Expected output: torch.Size([8, 256])
            ```
        """
        super(RetrievalModel, self).__init__()
        if backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True).features
            num_features = 1280 # Output of MobileNetV2 Convolutional Layer
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            # Remove AvgPool and FC
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            num_features = resnet.fc.in_features # Output of ResNet50 Convolutional Layer
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze all layers except the last 10
        for _, param in list(self.backbone.named_parameters())[:-10]:
            param.requires_grad = False

        # Embedding Layer
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Output shape: (batch_size, num_features, 1, 1)
            nn.Flatten(), # Output shape: (batch_size, num_features)
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        """
        Performs the forward pass of the RetrievalModel.

        This method processes the input tensor through the backbone network to extract high-level feature representations.
        The extracted features are then passed through the embedding layers to generate fixed-dimensional embedding vectors.

        Args:
            x (torch.Tensor): A batch of input images with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: A batch of embedding vectors with shape (batch_size, embedding_dim).

        Example:
            ```python
            # Initialize the model
            model = RetrievalModel(backbone='resnet50', embedding_dim=256)
            model.eval()

            # Create a dummy input tensor with batch size 8 and 3 color channels, 224x224 pixels
            input_tensor = torch.randn(8, 3, 224, 224)

            # Generate embeddings
            embeddings = model(input_tensor)
            print(embeddings.shape)  # Output: torch.Size([8, 256])
            ```
        """
        x = self.backbone(x)
        embedding = self.embedding(x)
        return embedding