import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
import sentencepiece

class LoraImageCaptioningModel(nn.Module):
    """
    Image captioning model that combines a CLIP vision encoder with Llama 3.2 1B decoder,
    using LoRA for efficient fine-tuning.
    
    The model takes image inputs and text prompts, and outputs text that responds to both.
    """
    def __init__(
        self,
        vision_model_name="openai/clip-vit-large-patch14",
        llm_model_name="meta-llama/Llama-3.2-1B",
        vision_embed_dim=1024,
        projection_dim=2048,  # Llama 3.2 1B hidden dimension
        max_length=512,
        lora_r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha scaling parameter
        lora_dropout=0.1,
        device=None
    ):
        super().__init__()
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CLIP vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        
        # Freeze the vision encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # Initialize Llama 3.2 1B decoder
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Configure LoRA for efficient fine-tuning of the language model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to the language model
        self.language_model = get_peft_model(self.language_model, lora_config)
        
        # Image embedding projection layer (maps CLIP's output to LLM's input space)
        self.image_projection = nn.Linear(vision_embed_dim, projection_dim)
        
        # Store configuration
        self.max_length = max_length
        self.projection_dim = projection_dim
        
        # Move to device
        self.to(self.device)
    
    def encode_image(self, pixel_values):
        """
        Encode images using the CLIP vision encoder.
        
        Args:
            pixel_values: Tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Image embeddings of shape (batch_size, projection_dim)
        """
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_embeds = vision_outputs.pooler_output  # [batch_size, vision_embed_dim]
        
        # Project to match LLM's embedding dimension
        image_embeds = self.image_projection(image_embeds)  # [batch_size, projection_dim]
        return image_embeds
    
    def prepare_inputs(self, images, prompts):
        """
        Prepare inputs for the model by encoding images and tokenizing prompts.
        
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            prompts: List of strings representing the text prompts
            
        Returns:
            tokenized_inputs: Dict containing input_ids and attention_mask
            image_embeds: Tensor of shape (batch_size, projection_dim)
        """
        # Encode images
        image_embeds = self.encode_image(images)
        
        # Tokenize prompts
        tokenized_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        return tokenized_inputs, image_embeds
    
    def forward(self, images, prompts, labels=None):
        """
        Forward pass of the model.
        
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            prompts: List of strings representing the text prompts
            labels: Optional tensor of shape (batch_size, seq_length) for training
            
        Returns:
            If labels are provided, returns the loss.
            Otherwise, returns logits of shape (batch_size, seq_length, vocab_size)
        """
        # Prepare inputs
        tokenized_inputs, image_embeds = self.prepare_inputs(images, prompts)
        
        # Extract inputs
        input_ids = tokenized_inputs.input_ids
        attention_mask = tokenized_inputs.attention_mask
        
        # Get llm embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)  
        
        # Prepend image embeddings to the first token embeddings of each sequence
        batch_size = inputs_embeds.shape[0]
        inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + image_embeds
        
        if labels is not None:
            # Training mode with teacher forcing
            
            # Create causal attention mask (explicit, though this happens internally in the model too)
            # This ensures that each token can only attend to previous tokens during training
            seq_length = input_ids.size(1)
            causal_mask = torch.tril(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=self.device)
            ).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]
            
            # Forward pass through the language model with teacher forcing
            # Note: The model will use the causal mask internally for the loss calculation
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,  # This activates teacher forcing and loss calculation
                return_dict=True
            )
            
            # Return loss for training
            return outputs.loss
        else:
            # Inference mode - no teacher forcing
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Return logits for inference
            return outputs.logits
    
    def generate(self, images, prompts, max_new_tokens=100, **generate_kwargs):
        """
        Generate text based on images and prompts.
        
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            prompts: List of strings representing the text prompts
            max_new_tokens: Maximum number of new tokens to generate
            generate_kwargs: Additional keyword arguments for generation
            
        Returns:
            List of generated text strings
        """
        # Prepare inputs
        tokenized_inputs, image_embeds = self.prepare_inputs(images, prompts)
        
        # Extract inputs
        input_ids = tokenized_inputs.input_ids
        attention_mask = tokenized_inputs.attention_mask
        
        # Get llm embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)  
        
        # Prepend image embeddings to the first token embeddings of each sequence
        inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + image_embeds
        
        # Note: During generation, the model internally applies causal masking
        # to ensure each new token only attends to previous tokens
        
        # Default generation parameters if not specified
        if 'do_sample' not in generate_kwargs and 'num_beams' not in generate_kwargs:
            generate_kwargs['num_beams'] = 3  # Use beam search by default
        
        # Generate text using the language model
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # Enable KV caching for faster generation
            **generate_kwargs
        )
        
        # Decode the generated token IDs to text
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        return generated_texts
    
    def save_pretrained(self, output_dir):
        """
        Save the model to the given directory.
        
        Args:
            output_dir: Path to save the model
        """
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save the vision encoder
        self.vision_encoder.save_pretrained(os.path.join(output_dir, "vision_encoder"))
        
        # Save the language model (with LoRA adapters)
        self.language_model.save_pretrained(os.path.join(output_dir, "language_model"))
        
        # Save the image projection layer
        torch.save(self.image_projection.state_dict(), os.path.join(output_dir, "image_projection.pt"))
        
        # Save model config
        import json
        config = {
            "vision_embed_dim": self.vision_encoder.config.hidden_size,
            "projection_dim": self.projection_dim,
            "max_length": self.max_length,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, model_dir):
        """
        Load the model from the given directory.
        
        Args:
            model_dir: Path to load the model from
            
        Returns:
            Loaded model
        """
        import os
        import json
        
        # Load config
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            vision_model_name=os.path.join(model_dir, "vision_encoder"),
            llm_model_name=os.path.join(model_dir, "language_model"),
            vision_embed_dim=config["vision_embed_dim"],
            projection_dim=config["projection_dim"],
            max_length=config["max_length"]
        )
        
        # Load image projection layer
        model.image_projection.load_state_dict(
            torch.load(os.path.join(model_dir, "image_projection.pt"))
        )
        
        return model

# # Example usage
# def example_usage():
#     """Example of how to use the model for training and inference"""
    
#     # Initialize model
#     model = LoraImageCaptioningModel()
    
#     # Example batch
#     batch_size = 4
#     image_size = 224
#     images = torch.randn(batch_size, 3, image_size, image_size).to(model.device)
#     prompts = ["Describe this image:", "What's happening in this picture?"] * 2
    
#     # Training example
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
#     # Forward pass for training (with dummy labels)
#     tokenized_labels = model.tokenizer(
#         ["This is a cat sitting on a mat", "A dog running in the park"] * 2,
#         padding="max_length",
#         max_length=model.max_length,
#         truncation=True,
#         return_tensors="pt"
#     ).input_ids.to(model.device)
    
#     loss = model(images, prompts, labels=tokenized_labels)
    
#     # Backward pass and optimization
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
    
#     # Inference example
#     generate_kwargs = {
#         "do_sample": True,
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "num_beams": 3,
#     }
    
#     generated_texts = model.generate(
#         images, 
#         prompts,
#         max_new_tokens=50,
#         **generate_kwargs
#     )
    
#     print("Generated captions:")
#     for prompt, text in zip(prompts, generated_texts):
#         print(f"Prompt: {prompt}")
#         print(f"Generated: {text}\n")
    
#     # Save model
#     model.save_pretrained("./saved_model")
    
#     # Load model
#     loaded_model = LoraImageCaptioningModel.from_pretrained("./saved_model")
    
#     return model, loaded_model

# if __name__ == "__main__":
#     example_usage()