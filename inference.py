import torch
from model import resnet101
from utils.create_transforms import create_inference_transform
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig

batch_size = 64
num_workers = 10
image_height = 224
image_width = 224
test_dir = "./data_val/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = resnet101(num_classes=4).eval()
    state_dict = torch.load("QAT_model.pth")
    model.load_state_dict(state_dict, assign=True)
    #model.load_state_dict(checkpoint["state_dict"])
    print("test")
    #print(model.fc)
    #model.to(device)

    inference_transform = create_inference_transform(image_height, image_width)
    inference_dataset = datasets.ImageFolder(root=test_dir, transform=inference_transform)
    inference_loader = DataLoader(inference_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            pin_memory=True)

    # Inference
    all_paths, all_labels, all_preds, all_probs = [], [], [], []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(inference_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)                    # logits
            probs = F.softmax(outputs, dim=1)          # probabilities
            preds = torch.argmax(probs, dim=1)         # predicted labels

            # Batch image paths
            batch_paths = [inference_dataset.samples[i + batch_idx * batch_size][0] 
                           for i in range(len(labels))]

            all_paths.extend(batch_paths)
            all_labels.extend(labels.cpu().numpy())    # keep as integers
            all_preds.extend(preds.cpu().numpy())      # keep as integers
            all_probs.extend(probs.cpu().numpy())

    # Build DataFrame
    df = pd.DataFrame({
        "path": all_paths,
        "actual_label": all_labels,
        "predicted_label": all_preds,
    })

    # Add probability columns
    prob_df = pd.DataFrame(all_probs, columns=[f"prob_class_{i}" for i in range(all_probs[0].shape[0])])
    df = pd.concat([df, prob_df], axis=1)

    df.to_csv("inference_results.csv", index=False)
    print("Saved predictions to inference_results.csv")

if __name__ == "__main__":
    main()