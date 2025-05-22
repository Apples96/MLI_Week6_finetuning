import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
# import sentencepiece

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
        max_length=128,
        lora_r=8,  # LoRA rank
        lora_alpha=16,  # LoRA alpha scaling parameter
        lora_dropout=0.1,
        device=None
    ):
        super().__init__()
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CLIP vision encoder
        self.image_processor = CLIPProcessor.from_pretrained(vision_model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        
        # Freeze the vision encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # Initialize Llama 3.2 1B decoder
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.loss_ignore_index = -100
        self.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
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
            image_embeds = image_embeds.unsqueeze(1)  # [batch_size, 1, vision_embed_dim]
        
        # Project to match LLM's embedding dimension
        image_embeds = self.image_projection(image_embeds)  # [batch_size, 1, projection_dim]
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
        image_inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        # print(f"prepare_inputs - image_inputs{image_inputs}")
        pixel_values = image_inputs['pixel_values'].to(self.device)
        # print(f"prepare_inputs - pixel_values.shape{pixel_values.shape}")
        # print(f"prepare_inputs - pixel_values{pixel_values}")
        image_embeds = self.encode_image(pixel_values) # [batch_size, projection_dim]
        # image_embeds = image_embeds.unsqueeze(1) # [batch_size, seq_length (ie 1 pooled vector), projection_dim]
        
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
        # print(f"images.shape{images.shape}")
        # print(f"images{images}")
        # print(f"prompts.shape : no shape bc list")
        # print(f"prompts{prompts}")
        # print(f"labels.shape{labels.shape}")
        # print(f"labels{labels}")


        
        # Prepare inputs
        tokenized_inputs, image_embeds = self.prepare_inputs(images, prompts)
        # print("=======================================")
        # print(f"image_embeds.shape{image_embeds.shape}")
        # print(f"image_embeds{image_embeds}")
        # print(f"prompts tokenized_inputs.shape : None bc multiple inputs, not just token ids")
        # print(f"prompts tokenized_inputs{tokenized_inputs}")
        
        # Extract inputs
        input_ids = tokenized_inputs.input_ids
        attention_mask = tokenized_inputs.attention_mask
        # print(f"prompts input_ids.shape{input_ids.shape}")
        # print(f"prompts input_ids{input_ids}")
        # print(f"prompts attention_mask.shape{attention_mask.shape}")
        # print(f"prompts attention_mask{attention_mask}")
        
        # Get llm embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)  
        # print("=======================================")
        # print(f"prompts inputs_embeds.shape{inputs_embeds.shape}")
        # print(f"prompts inputs_embeds{inputs_embeds}")
        
        # Prepend image embeddings as separate tokens
        batch_size = inputs_embeds.shape[0]
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)  # [batch_size, 1+seq_length, hidden_size]
        batch_size, seq_length, hidden_size = inputs_embeds.shape

        # Extend the attention mask
        image_attention = torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        # print("=======================================")
        # print(f"batch_size{batch_size}")
        # print(f"prompts + image inputs_embeds.shape{inputs_embeds.shape}")
        # print(f"prompts + image inputs_embeds{inputs_embeds}")
        # print(f"prompts + image attention_mask.shape{attention_mask.shape}")
        # print(f"prompts + image attention_mask{attention_mask}")
        
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(0)
            # Training mode with teacher forcing
            new_labels = torch.full((batch_size, 1 + labels.size(1)), self.loss_ignore_index, 
                                    device=self.device, dtype=labels.dtype)
            # print(f"new_labels.shape{new_labels.shape}")
            # print(f"new_labels{new_labels}")
            new_labels[:, 1:] = labels
            # Then mask padding tokens
            padding_mask = new_labels == self.tokenizer.pad_token_id
            new_labels[padding_mask] = self.loss_ignore_index


            # Create causal attention mask (explicit, though this happens internally in the model too)
            # This ensures that each token can only attend to previous tokens during training
            seq_length = input_ids.size(1)
            causal_mask = torch.tril(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=self.device)
            ).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]
            # print(f"for show : prompts causal_mask.shape{causal_mask.shape}")
            # print(f"for show : prompts causal_mask{causal_mask}")
            
            # Forward pass through the language model with teacher forcing
            # Note: The model will use the causal mask internally for the loss calculation
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=new_labels,  # This activates teacher forcing and loss calculation
                return_dict=True
            )
            # print("=======================================")
            # print(f"outputs.shape : no shape bc 'CausalLMOutputWithPast' object has no attribute 'shape'")
            # print(f"outputs{outputs}")
            
            # Return loss for training
            # print(f"outputs.logits.shape{outputs.logits.shape}")
            # print(f"outputs.logits{outputs.logits}")
            # print(f"outputs.loss.shape{outputs.loss.shape}")
            # print(f"outputs.loss{outputs.loss}")
            return outputs.loss
        else:
            # Inference mode - no teacher forcing
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Return logits for inference
            # print(f"outputs.logits.shape{outputs.logits.shape}")
            # print(f"outputs.logits{outputs.logits}")
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
        
        # Prepend image embeddings as separate tokens
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)  # [batch_size, 1+seq_length, hidden_size]

        # Extend the attention mask
        batch_size = inputs_embeds.shape[0]
        image_attention = torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
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
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
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
