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

# Local imports
from data import Flickr30k
from models import image_captioning_model, lora_image_captioning_model


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
        num_workers=2,       # Adjust based on your CPU cores
        pin_memory=True      # Speeds up CPU to GPU transfers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model and move to device
    model = lora_image_captioning_model().to(device)

    # Set up optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # CrossEntropyLoss with 'none' reduction to apply masking for padding tokens
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

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
            # Zero gradients
            optimizer.zero_grad()

            # Get batch data
            images = batch["image"].to(device)
            captions = batch["caption"]

            # Forward pass
            all_logits, caption_logits, tokenized_images, prompt_input_ids, prompt_target_ids, extended_input_attention_mask, extended_target_attention_mask, prompt_target_attention_mask = model(images, captions)
            
            # Print debug info for first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"First batch shapes:")
                print(f"  Images: {tokenized_images.shape}")
                print(f"  Input caption IDs: {prompt_input_ids.shape}")
                print(f"  Target caption IDs: {prompt_target_ids.shape}")
                print(f"  Caption logits: {caption_logits.shape}")
                assert caption_logits.shape[1] == prompt_target_ids.shape[1], (
                    f"Caption logits shape {caption_logits.shape} and target IDs shape "
                    f"{prompt_target_ids.shape} don't match in sequence length dimension"
                )

            # Reshape for loss calculation
            reshaped_logits = caption_logits.reshape(-1, model.vocab_size)  # [batch_size * seq_len, vocab_size]
            reshaped_targets = prompt_target_ids.reshape(-1)                # [batch_size * seq_len]

            # Calculate loss
            loss = criterion(reshaped_logits, reshaped_targets)

            # Apply mask to ignore padding tokens
            if extended_target_attention_mask is not None:
                # Get the proper text attention mask from the original prompt
                # We don't need to use extended_target_attention_mask here because
                # we're only masking the loss for the text part (caption_logits)
                text_mask = prompt_target_attention_mask.reshape(-1).float()
                
                # Verify shapes match
                assert text_mask.shape[0] == reshaped_targets.shape[0], (
                    f"Text mask shape {text_mask.shape} doesn't match targets shape {reshaped_targets.shape}"
                )
                
                # Zero out loss for padding tokens
                masked_loss = loss * text_mask
                
                # Average loss over non-padding positions
                loss = masked_loss.sum() / (text_mask.sum() + 1e-8)
            else:
                loss = loss.mean()

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

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get batch data
                images = batch["image"].to(device)
                captions = batch["caption"]
                
                # Forward pass
                all_logits, caption_logits, tokenized_images, prompt_input_ids, prompt_target_ids, extended_input_attention_mask, extended_target_attention_mask, prompt_target_attention_mask = model(images, captions)
                
                # Reshape for loss calculation
                reshaped_logits = caption_logits.reshape(-1, model.vocab_size)
                reshaped_targets = prompt_target_ids.reshape(-1)
                
                # Calculate loss
                loss = criterion(reshaped_logits, reshaped_targets)
                
                # Apply mask to ignore padding tokens
                if extended_target_attention_mask is not None:
                    # Get the proper text attention mask from the original prompt
                    text_mask = prompt_target_attention_mask.reshape(-1).float()
                    
                    # Verify shapes match
                    assert text_mask.shape[0] == reshaped_targets.shape[0], (
                        f"Text mask shape {text_mask.shape} doesn't match targets shape {reshaped_targets.shape}"
                    )
                    
                    # Zero out loss for padding tokens
                    masked_loss = loss * text_mask
                    loss = masked_loss.sum() / (text_mask.sum() + 1e-8)
                else:
                    loss = loss.mean()
                
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
            sample_caption = model.generate(sample_image, ["Describe this image:"])
            print(f"Sample generated caption: {sample_caption[0]}")
            wandb.log({"sample_caption": sample_caption[0]})
        except Exception as e:
            print(f"Error generating sample caption: {e}")
    
    # Final cleanup
    wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    # This will run when the script is executed directly
    train_lora_image_caption(
        batch_size=2,
        num_epochs=5,
        learning_rate=1e-4, 
        max_samples=10  # Use small sample for quick testing
    )