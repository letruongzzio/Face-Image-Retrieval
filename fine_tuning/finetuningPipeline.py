import os
import torch
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from triplet_method import TripletDataset, QueryDataset, collate_fn
from triplet_process import GenerateTriplets
from retrieval_models import RetrievalModel
from model_finetuning import fine_tune_with_identity
from compute_embedding_celebA import compute_embeddings_from_images
from drop_out import RandomDropout

# Configurations
PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
IMAGE_DIR = os.path.join(PARENT_DIRNAME, "data/img_align_celeba/")
IDENTITY_FILE_PATH = os.path.join(PARENT_DIRNAME, "data/identity_CelebA.txt")
MODEL_DIR = os.path.join(PARENT_DIRNAME, "fine_tuning/models/")
STORAGE_DATA_DIRNAME = os.path.join(PARENT_DIRNAME, "fine_tuning/data_for_fine_tuning")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
EMBEDDING_DIM = 128
LEARNING_RATE = 1e-4
MARGIN = 1.0
NUM_THREADS = 4
IMAGE_SIZE = 218
BATCH_SIZE = [1, 32, 64]
NUM_WORKERS = 4

# Load Identity Labels
if not os.path.exists(os.path.join(STORAGE_DATA_DIRNAME, "train_triplets.json")):
    identity_labels = GenerateTriplets(num_threads=NUM_THREADS)
    train_triplets, test_queries, test_galleries = identity_labels

    with open(os.path.join(STORAGE_DATA_DIRNAME, "train_triplets.json"), "w", encoding="utf-8") as f:
        json.dump(train_triplets, f)

    with open(os.path.join(STORAGE_DATA_DIRNAME, "test_queries.json"), "w", encoding="utf-8") as f:
        json.dump(test_queries, f)

    with open(os.path.join(STORAGE_DATA_DIRNAME, "test_galleries.json"), "w", encoding="utf-8") as f:
        json.dump(test_galleries, f)

else:
    with open(os.path.join(STORAGE_DATA_DIRNAME, "train_triplets.json"), "r", encoding="utf-8") as f:
        train_triplets = json.load(f)

    with open(os.path.join(STORAGE_DATA_DIRNAME, "test_queries.json"), "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    with open(os.path.join(STORAGE_DATA_DIRNAME, "test_galleries.json"), "r", encoding="utf-8") as f:
        test_galleries = json.load(f)

# Load DataLoaders
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    transforms.RandomApply([RandomDropout(p=0.3)], p=0.3)
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
if not os.path.exists(os.path.join(STORAGE_DATA_DIRNAME, "train_loader.pth")):
    train_dataset = TripletDataset(
        image_dir=IMAGE_DIR,
        train_triplets=train_triplets,
        transform=train_transforms
    )

    query_dataset = QueryDataset(
        image_dir=IMAGE_DIR,
        query_triplets=test_queries,
        transform=test_transforms
    )

    gallery_dataset = QueryDataset(
        image_dir=IMAGE_DIR,
        query_triplets=test_galleries,
        transform=test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[2], shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE[0], shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE[1], shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    torch.save(train_loader, os.path.join(STORAGE_DATA_DIRNAME, "train_loader.pth"))
    torch.save(query_loader, os.path.join(STORAGE_DATA_DIRNAME, "query_loader.pth"))
    torch.save(gallery_loader, os.path.join(STORAGE_DATA_DIRNAME, "gallery_loader.pth"))

else:
    train_loader = torch.load(os.path.join(STORAGE_DATA_DIRNAME, "train_loader.pth"))
    query_loader = torch.load(os.path.join(STORAGE_DATA_DIRNAME, "query_loader.pth"))
    gallery_loader = torch.load(os.path.join(STORAGE_DATA_DIRNAME, "gallery_loader.pth"))

# Define the Model
backbone = input("Enter the backbone model (resnet50, mobilenet_v2): ")
model = RetrievalModel(backbone=backbone, embedding_dim=EMBEDDING_DIM).to(DEVICE)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Fine-tuning the Model
fine_tune_with_identity(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=DEVICE,
    num_threads=NUM_WORKERS
)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{backbone}_identity.pth"))

# Calculate Embeddings
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{backbone}_identity.pth")))

full_embeddings, full_labels = compute_embeddings_from_images(
    model=model,
    device=DEVICE
)

torch.save(full_embeddings, os.path.join(STORAGE_DATA_DIRNAME, f"full_embeddings_{backbone}.pth"))
torch.save(full_labels, os.path.join(STORAGE_DATA_DIRNAME, f"full_labels_{backbone}.pth"))
