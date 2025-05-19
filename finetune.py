import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from PIL import Image

# class website_captioning(Dataset):
#     def init(self):
#         XXXX
#     def get_item(idx):





# Load CLIP ViT encoder model

class image_captioning_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = self.clip.vision_model # gets the output logits from the final layer of the encoder, but before any final softmax is applied

        self.text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.text_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

        self.encoder_projection = nn.Linear(   # projection layer that passes the encoder output hidden states of dimension X into the decoder input vector size Y after the embedding layer
            self.vision_encoder.config.hidden_size,  # CLIP's hidden size
            self.text_model.config.hidden_size       # Llama's hidden size
        ) 

        # Pseudo code V1 - to be deleted. 
        # self.text_embedding = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").embedding_layer # takes only the embedding layer from Llama-3.2-1B-Instruct
        # self.decoder_model_from_embeddings_to_hidden_output = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").whole_model_starting_after_embedding_layer # takes all layers after the embedding layer from Llama-3.2-1B-Instruct, so that we can preprend the iamge logits just after the embedding layer. 
        # self.decoder_model_from_hidden_outputs_to_final_outputs = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").very_last_layer_to_tokens
    
    
    def forward(self, images, prompts):
        # Get image encoder outputs and project to text embed dimension
        image_inputs = self.image_processor(images=images, return_tensors="pt")
        image_features = self.vision_encoder(**image_inputs).last_hidden_state # Note : ** before a variable in a function call is the "dictionary unpacking" operator.? Equivalent to self.vision_encoder(pixel_values=tensor_data, attention_mask=mask_data)
        projected_image_features = self.encoder_projection(image_features)
        image_attention_mask = torch.ones_like(projected_image_features) # creates a matrix size projected_image_features filled with ones, so that all of these are attended to when fed into the decoder. 

        # Get prompt embeddings and attention mask - no ned to sadd a special token  to identify end of image and stat of text, bc tokenizer adds a BOS token to start of prompt. 
        prompt_inputs = self.text_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        prompt_input_ids = prompt_inputs.input_ids
        prompt_attention_mask = prompt_inputs.attention_mask

        prompts_embeddings = self.text_model.get_input_embeddings()(prompt_input_ids)

        # Add image and text embeddings and attention masks to get extended inputs. Note = we don't need rotary positional encoding as the model will handle this internally
        extended_embedding = torch.cat([projected_image_features, prompts_embeddings], dim=1) # this should be of (batch_size, (image_num_patches + prompt_num_tokens), dimension text_embedding_dim) 
        extended_attention_mask = torch.cat([image_attention_mask, prompt_attention_mask], dim=1) 

        outputs = self.text_model(
            inputs_embeds=extended_embedding,
            attention_mask=extended_attention_mask,
            return_dict=True
        )

        return outputs

    def generate(self, images, prompts, max_length=50):
        # My forst attempt, likely flawed
        # logits = self.forward(images, prompts).last_hidden_state
        # # Use the embedding weights transposed for the final projection - double check this is what Llama does. Ideally would use the generate mode in Hugging Face. 
        # output_ids = self.text_model.lm_head(logits) # Should be of dim (batch_size, num_tokens, 128257) > vocab is 128256 + 1 new image token
        # captions = self.text_tokenizer.batch_decode(
        #     output_ids.sequences, 
        #     skip_special_tokens=True
        # )

        # return captions
    
        # Claude suggestion for the generate function : 
        #Process images
        image_inputs = self.image_processor(images=images, return_tensors="pt")
        image_features = self.vision_encoder(**image_inputs).last_hidden_state
        projected_image_features = self.encoder_projection(image_features)
        
        # Process prompts for initial conditioning
        prompt_inputs = self.text_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Create input IDs with image token
        batch_size = prompt_inputs.input_ids.shape[0]
        image_token_tensor = torch.tensor([[self.image_token_id]] * batch_size, device=prompt_inputs.input_ids.device)
        prompt_input_ids = torch.cat([image_token_tensor, prompt_inputs.input_ids], dim=1)
        
        # Create proper attention mask
        image_seq_length = projected_image_features.shape[1]
        image_attention_mask = torch.ones(batch_size, image_seq_length, device=prompt_inputs.input_ids.device)
        image_token_attention = torch.ones(batch_size, 1, device=prompt_inputs.attention_mask.device)
        prompt_attention_mask = torch.cat([image_token_attention, prompt_inputs.attention_mask], dim=1)
        extended_attention_mask = torch.cat([image_attention_mask, prompt_attention_mask], dim=1)
        
        # Get embeddings
        prompts_embeddings = self.text_model.get_input_embeddings()(prompt_input_ids)
        
        # Combine embeddings
        extended_embedding = torch.cat([projected_image_features, prompts_embeddings], dim=1)
        
        # Use the model's generate functionality to create sequences
        # Note: This is simplified and would need a custom generate implementation
        # for proper conditioning on the image embeddings
        
        with torch.no_grad():
            output_ids = self.text_model.generate(
                inputs_embeds=extended_embedding,
                attention_mask=extended_attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode the generated sequences
        captions = self.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return captions
    


if __name__ == "__main__":
    model = image_captioning_model()
    print(model)

    












