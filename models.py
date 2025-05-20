"""
Image captioning models using CLIP visual encoder and Llama text generation.
Two implementations:
1. Base model with full fine-tuning
2. LoRA version for parameter-efficient fine-tuning
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


class image_captioning_model(nn.Module):
    """
    Image captioning model that combines a CLIP vision encoder with a Llama text decoder.
    The model processes images with CLIP, projects the features to the text model's space,
    and generates captions using the Llama language model.
    """
    def __init__(self):
        super().__init__()

        # Vision components: CLIP for image encoding
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = self.clip.vision_model  # Just the vision part, no text

        # Text components: Llama for text generation
        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token  # Use EOS as pad token
        self.text_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.vocab_size = self.text_model.config.vocab_size

        # Projection layer to map CLIP features to Llama embedding space
        self.encoder_projection = nn.Linear(
            self.vision_encoder.config.hidden_size,  # CLIP's hidden dimension
            self.text_model.config.hidden_size       # Llama's hidden dimension
        )
    
    def forward(self, images, prompts):
        """
        Forward pass for training.
        
        Args:
            images: Batch of PIL images
            prompts: Text prompts to guide caption generation
            
        Returns:
            all_logits: Model logits for all tokens
            caption_logits: Model logits just for the caption part (excluding image tokens)
        """
        # 1. Process images with CLIP
        image_inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        image_inputs = {k: v.to(next(self.parameters()).device) for k, v in image_inputs.items()}
        tokenized_images = image_inputs['pixel_values']
        
        # 2. Get image features and project to text embedding dimension
        image_features = self.vision_encoder(**image_inputs).last_hidden_state
        projected_image_features = self.encoder_projection(image_features)
        
        # 3. Create attention mask for image tokens (all 1s)
        batch_size = projected_image_features.shape[0]
        seq_length = projected_image_features.shape[1]
        image_attention_mask = torch.ones(
            (batch_size, seq_length), 
            dtype=torch.long, 
            device=projected_image_features.device
        )

        # 4. Tokenize and embed text prompts
        prompt_inputs = self.text_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='max_length', 
            max_length=100,
            truncation=True
        )
        prompt_input_ids = prompt_inputs.input_ids.to(projected_image_features.device)
        prompt_attention_mask = prompt_inputs.attention_mask.to(projected_image_features.device)
        prompts_embeddings = self.text_model.get_input_embeddings()(prompt_input_ids)

        # 5. Concatenate image and text embeddings and attention masks
        extended_embedding = torch.cat([projected_image_features, prompts_embeddings], dim=1)
        extended_attention_mask = torch.cat([image_attention_mask, prompt_attention_mask], dim=1)

        # 6. Run Llama model on the combined sequence
        outputs = self.text_model(
            inputs_embeds=extended_embedding,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        
        # 7. Get logits for full sequence and just the caption part
        all_logits = outputs.logits
        caption_logits = outputs.logits[:, seq_length:, :]

        return all_logits, caption_logits

    def generate(self, images, prompts, max_new_tokens=50):
        """
        Generate captions for images.
        
        Args:
            images: List of PIL images
            prompts: Text prompts to guide caption generation
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of generated captions
        """
        device = next(self.parameters()).device
        
        # 1. Process images with CLIP
        image_inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_features = self.vision_encoder(**image_inputs).last_hidden_state
        projected_image_features = self.encoder_projection(image_features)
        
        # 2. Create attention mask for image tokens
        batch_size = projected_image_features.shape[0]
        seq_length = projected_image_features.shape[1]
        image_attention_mask = torch.ones(
            (batch_size, seq_length), 
            dtype=torch.long, 
            device=device
        )
        
        # 3. Process prompts
        prompt_inputs = self.text_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        prompt_input_ids = prompt_inputs.input_ids.to(device)
        prompt_attention_mask = prompt_inputs.attention_mask.to(device)
        
        # 4. Combine image and text information
        extended_attention_mask = torch.cat([image_attention_mask, prompt_attention_mask], dim=1)
        prompts_embeddings = self.text_model.get_input_embeddings()(prompt_input_ids)
        extended_embedding = torch.cat([projected_image_features, prompts_embeddings], dim=1)
        
        # 5. Generate text with the model
        with torch.no_grad():
            output_ids = self.text_model.generate(
                inputs_embeds=extended_embedding,
                attention_mask=extended_attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True, 
                pad_token_id=self.text_tokenizer.pad_token_id
            )
        
        # 6. Decode the generated sequences
        captions = self.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return captions


class lora_image_captioning_model(nn.Module):
    """
    LoRA-based image captioning model for parameter-efficient fine-tuning.
    Uses the same architecture as image_captioning_model but applies LoRA adapters
    to specific layers in the text model.
    """
    def __init__(self):
        super().__init__()
        
        # Vision components
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = self.clip.vision_model
        
        # Text components
        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.vocab_size = self.text_model.config.vocab_size
        
        # Projection layer
        self.encoder_projection = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.text_model.config.hidden_size
        )
        
        # LoRA configuration
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=2,               # LoRA attention dimension (rank)
            lora_alpha=32,     # LoRA alpha parameter
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            bias="none",       # Whether to train bias parameters
            # Which layers to apply LoRA to
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
            ],
        )
        
        # Apply LoRA to the text model
        self.lora_text_model = get_peft_model(self.text_model, self.peft_config)
    
    def forward(self, images, prompts):
        """
        Forward pass for training with LoRA.
        
        Args:
            images: Batch of PIL images
            prompts: Text prompts to guide caption generation
            
        Returns:
            all_logits: Model logits for all tokens
            caption_logits: Model logits just for the caption part
            Additional tensors needed for training
        """
        device = next(self.parameters()).device
        
        # 1. Process images with CLIP
        image_inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        tokenized_images = image_inputs['pixel_values']
        image_features = self.vision_encoder(**image_inputs).last_hidden_state
        projected_image_features = self.encoder_projection(image_features)
        
        # 2. Create attention mask for image tokens
        batch_size = projected_image_features.shape[0]
        seq_length = projected_image_features.shape[1]
        image_attention_mask = torch.ones(
            (batch_size, seq_length), 
            dtype=torch.long, 
            device=device
        )

        # 3. Process text prompts
        prompt_inputs = self.text_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='max_length', 
            max_length=100,
            truncation=True
        )
            
        # 4. Prepare input and target sequences for teacher forcing
        prompt_ids = prompt_inputs.input_ids.to(device)
        prompt_input_ids = prompt_ids[:, :-1]  # Input: all tokens except last
        prompt_target_ids = prompt_ids[:, 1:]  # Target: all tokens except first
        prompt_input_attention_mask = prompt_inputs.attention_mask[:, :-1].to(device)
        prompt_target_attention_mask = prompt_inputs.attention_mask[:, 1:].to(device)
        prompts_embeddings = self.text_model.get_input_embeddings()(prompt_input_ids)

        # 5. Combine image and text information
        extended_embedding = torch.cat([projected_image_features, prompts_embeddings], dim=1)
        extended_input_attention_mask = torch.cat([image_attention_mask, prompt_input_attention_mask], dim=1)
        extended_target_attention_mask = torch.cat([image_attention_mask, prompt_target_attention_mask], dim=1)

        # 6. Run LoRA-adapted Llama model
        outputs = self.lora_text_model(
            inputs_embeds=extended_embedding,
            attention_mask=extended_input_attention_mask,
            return_dict=True
        )
        
        # 7. Get logits
        all_logits = outputs.logits
        caption_logits = all_logits[:, seq_length:, :]

        # Return everything needed for training
        return (
            all_logits, 
            caption_logits, 
            tokenized_images, 
            prompt_input_ids, 
            prompt_target_ids, 
            extended_input_attention_mask, 
            extended_target_attention_mask, 
            prompt_target_attention_mask
        )
    
    def generate(self, images, prompts, max_new_tokens=50):
        """
        Generate captions using the LoRA-adapted model.
        
        Args:
            images: List of PIL images
            prompts: Text prompts to guide caption generation
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of generated captions
        """
        device = next(self.parameters()).device
        
        # 1. Process images with CLIP
        image_inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_features = self.vision_encoder(**image_inputs).last_hidden_state
        projected_image_features = self.encoder_projection(image_features)
        
        # 2. Create attention mask for image tokens
        batch_size = projected_image_features.shape[0]
        seq_length = projected_image_features.shape[1]
        image_attention_mask = torch.ones(
            (batch_size, seq_length), 
            dtype=torch.long, 
            device=device
        )
        
        # 3. Process prompts
        prompt_inputs = self.text_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        prompt_input_ids = prompt_inputs.input_ids.to(device)
        prompt_attention_mask = prompt_inputs.attention_mask.to(device)
        
        # 4. Combine image and text information
        extended_attention_mask = torch.cat([image_attention_mask, prompt_attention_mask], dim=1)
        prompts_embeddings = self.text_model.get_input_embeddings()(prompt_input_ids)
        extended_embedding = torch.cat([projected_image_features, prompts_embeddings], dim=1)
        
        # 5. Generate text with the LoRA-adapted model
        with torch.no_grad():
            output_ids = self.lora_text_model.generate(
                inputs_embeds=extended_embedding,
                attention_mask=extended_attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True, 
                pad_token_id=self.text_tokenizer.pad_token_id
            )
        
        # 6. Decode the generated sequences
        captions = self.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return captions


if __name__ == "__main__":
    """Basic test of model functionality"""
    model = lora_image_captioning_model()
    print(model)

    # Test with a sample image and prompt
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Load a test image
    response = requests.get("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/dog.jpg")
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Generate a caption
    captions = model.generate([image], ["Describe this image:"])
    print(f"Generated caption: {captions[0]}")