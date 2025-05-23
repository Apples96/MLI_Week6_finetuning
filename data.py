"""
Flickr30k dataset loader.
Simple and efficient implementation for image captioning tasks.
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class Flickr30k(Dataset):
    """
    Memory-efficient Flickr30k dataset that loads images and captions on demand.
    
    Each item in the dataset is a (image, caption) pair. The dataset contains
    multiple captions per image, so we create a mapping to access each pair.
    """
    def __init__(self, split='test'):
        """
        Args:
            split: Dataset split ('train', 'test', 'validation')
        """
        print(f"Initializing Flickr30k dataset ({split} split)...")
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)
        
        # Standard image transformation pipeline for vision models
        self.transform = transforms.Compose([
            transforms.Resize(256),      # Resize to slightly larger than target
            transforms.CenterCrop(224),  # Crop to standard 224x224 for most vision models
            transforms.ToTensor()        # Convert to tensor and scale to [0, 1]
        ])
        
        print(f"Dataset initialized with {len(self.dataset)} image-caption pairs")
    
    def __len__(self):
        """Return the total number of image-caption pairs."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieve an image-caption pair by index.
        
        Args:
            idx: Global index in the dataset
            
        Returns:
            dict with keys:
                - 'image': Transformed image tensor
                - 'caption': Caption string
        """
        
        # Retrieve the item from the dataset
        item = self.dataset[idx]
        image = item['image']
        caption = item['caption'][0]
        
        # Apply transformation to image
        image = self.transform(image)
        
        return {
            "image": image,
            "caption": caption
        }