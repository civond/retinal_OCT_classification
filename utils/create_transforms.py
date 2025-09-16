import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
#from utils.get_loader import get_loader


# Create transforms object
def create_train_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # grayscale
    ])
        
    return transform

def create_valid_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    return transform

def create_inference_transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    return transform