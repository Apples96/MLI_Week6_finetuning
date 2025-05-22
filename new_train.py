"""
Training script for image captioning models.
Supports both full fine-tuning and LoRA training.
"""
import os
import torch
import random
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim import Adam
import wandb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import huggingface_hub


# Local imports
from data import Flickr30k
from new_models import LoraImageCaptioningModel


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_dataset(dataset, max_samples=100, random_seed=42):
    """
    Create a smaller subset of the dataset for faster iterations.
    
    Args:
        dataset: Dataset to sample from
        max_samples: Maximum number of samples to take
        random_seed: Seed for reproducibility
        
    Returns:
        Subset of the original dataset
    """
    if len(dataset) <= max_samples:
        return dataset
    
    # Set seed for reproducibility
    random.seed(random_seed)

    # Sample random indices
    indices = random.sample(range(len(dataset)), max_samples)
    return Subset(dataset, indices)


def train_lora_image_caption(
    batch_size=2,
    num_epochs=5,
    learning_rate=1e-4, 
    max_samples=10,
    save_dir="models"
):
    """
    Train an image captioning model with LoRA fine-tuning.
    
    Args:
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        max_samples: If provided, limit training to this many samples
        save_dir: Directory to save model checkpoints
    """
    set_seed(42)
    
    # Setup device - use MPS for Apple Silicon if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load dataset (or subset for quicker iteration)
    if max_samples is not None:
        dataset = sample_dataset(Flickr30k(), max_samples=max_samples)
        print(f"Using {max_samples} samples from Flickr30k")
    else:
        dataset = Flickr30k()
    
    # Split into train and validation sets (80% train, 20% val)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,       # Adjust based on your CPU cores
        pin_memory=True      # Speeds up CPU to GPU transfers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model and move to device
    model = LoraImageCaptioningModel(device=device)

    # Set up optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # CrossEntropyLoss is already included within the model and HuggingFace loss calculation

    padding_token_id = model.tokenizer.pad_token_id

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="lora_image-captioning", 
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "device": str(device)
        }
    )

    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in progress_bar:
            batch_size = batch['image'].shape[0]
            # print(f"batch_size{batch_size}")
            # Zero gradients
            optimizer.zero_grad()

            # Get batch data
            images = batch["image"].to(device)
            prompts = ["What is in this picture ?"]*batch_size
            captions = batch["caption"]
            # print(f"train.py images.shape{images.shape}")
            # print(f"train.py images{images}")
            # print(f"train.py prompts.len{len(prompts)} : no shape bc list")
            # print(f"train.py prompts{prompts}")
            # print(f"train.py captions.len{len(captions)} : no shape bc list")
            # print(f"train.py captions{captions}")

            tokenized_labels = model.tokenizer(
                captions,
                padding="max_length",
                max_length=model.max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            tokenized_labels[tokenized_labels == padding_token_id] = model.loss_ignore_index

            # Forward pass
            loss = model(images, prompts, labels=tokenized_labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            batch_loss = loss.item()
            train_loss += batch_loss
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"train_loss": batch_loss})
            
            # Log to W&B every 10 batches
            if batch_idx % 10 == 0:
                wandb.log({"batch_loss": batch_loss, "step": epoch * len(train_loader) + batch_idx})

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    debug_caption = model.generate(images[0].unsqueeze(0), ["What is in this image ?"])
                    print(f"Debug caption: {debug_caption[0]}")

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get batch data
                batch_size = batch['image'].shape[0]
                # print(f"batch_size{batch_size}")
                images = batch["image"].to(device)
                prompts = ["What is in this picture ?"]*batch_size
                captions = batch["caption"]
                
                # Forward pass
                tokenized_labels = model.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=model.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)
                # tokenized_labels[tokenized_labels == padding_token_id] = model.loss_ignore_index >> this doesn't work

                # Forward pass
                loss = model(images, prompts, labels=tokenized_labels)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batches
        
        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} complete: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Create a more detailed checkpoint with metadata
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save checkpoint
            torch.save(checkpoint, os.path.join(save_dir, "best_captioning_model.pt"))
            print(f"✓ New best model saved with val_loss: {best_val_loss:.4f}")
        
    # Generate a sample caption with the trained model
    if len(val_dataset) > 0:
        try:
            print("\nGenerating a sample caption with the trained model...")
            sample_batch = next(iter(val_loader))
            sample_image = sample_batch["image"][0:1].to(device)  # Take first image
            sample_caption = model.generate(sample_image, ["What is in this picture ?"])
            print(f"Sample generated caption: {sample_caption[0]}")
            wandb.log({"sample_caption": sample_caption[0]})
        except Exception as e:
            print(f"Error generating sample caption: {e}")
    
    # Final cleanup
    wandb.finish()
    print("Training completed!")


    # Test with a sample image and prompt
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Load a test image
    images_dir = "images"  # Directory containing your PNG images
    image_filename = "dogs.png"  # Replace with your actual image filename
    image_path = os.path.join(images_dir, image_filename)
    image = Image.open(image_path).convert("RGB")
    
    # Generate a caption
    captions = model.generate([image], ["Describe this image:"])
    print(f"Generated caption: {captions[0]}")


if __name__ == "__main__":
    # This will run when the script is executed directly
    train_lora_image_caption(
        batch_size=8,
        num_epochs=8,
        learning_rate=3e-5, 
        max_samples=10000  # Use small sample for quick testing
    )
