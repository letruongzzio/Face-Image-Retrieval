import torch.nn as nn
from torchvision import models

class RetrievalModel(nn.Module):
    """
    A retrieval model that extracts embeddings and predicts attributes.
    """
    def __init__(self, backbone='mobilenet_v2', embedding_dim=128, num_attributes=40):
        """
        Args:
            - backbone: Backbone architecture for the model.
            - embedding_dim: Dimension of the embedding.
            - num_attributes: Number of attributes to predict.
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

        # Attributes Classifier
        self.attributes_classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_attributes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = self.backbone(x)
        embedding = self.embedding(x)
        attributes = self.attributes_classifier(embedding)
        return embedding, attributes