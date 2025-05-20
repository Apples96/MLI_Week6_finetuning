import streamlit as st
import torch
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from models import image_captioning_model
import time

# Set page config
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the image captioning model and necessary processor"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and initla weights
    model = image_captioning_model().to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Load processor
    image_processor = image_captioning_model.image_processor
    text_tokenizer = image_captioning_model.text_tokenizer
    
    return model, image_processor, text_tokenizer, device

def generate_caption(model, image_processor, text_tokenizer, image, device, max_length=50):
    """Generate a caption for the given image"""
    # Process image with CLIP processor
    image_inputs = image_processor(
        images=image,
        return_tensors="pt",
    )
    
    # Move inputs to device
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    
    # Get image features
    tokenized_images = image_inputs["pixel_values"]
    
    # Create a properly padded sequence for the initial input
    # Start with just the BOS token
    generated_ids = [text_tokenizer.bos_token_id]
    
    # We need to match the expected text_seq_length from our model (76 tokens)
    # The model expects this to be padded with padding tokens
    # text_seq_length = model.text_seq_length  # Should be 76
    
    # Generate tokens auto-regressively
    with torch.no_grad():
        captions = model.generate(images, prompts, max_length=50)
        # for _ in range(max_length):
        #     # Pad the current sequence to the right length
        #     current_length = len(generated_ids)
        #     padding_length = text_seq_length - current_length
            
        #     # Create padded input_ids for this step
        #     padded_input_ids = generated_ids + [tokenizer.pad_token_id] * padding_length
        #     input_ids = torch.tensor([padded_input_ids[:text_seq_length]]).to(device)
            
        #     # Create a padding mask where real tokens are 1, padding tokens are 0
        #     padding_mask = torch.ones(1, current_length).to(device)
        #     if padding_length > 0:
        #         padding_zeros = torch.zeros(1, padding_length).to(device)
        #         padding_mask = torch.cat([padding_mask, padding_zeros], dim=1)
            
        #     # Forward pass
        #     output_logits = model(tokenized_images, input_ids, padding_mask)
            
        #     # Get the prediction for the last REAL token (not padding)
        #     next_token_logits = output_logits[0, current_length-1, :]
            
        #     # Get most likely token
        #     next_token_id = torch.argmax(next_token_logits).item()
            
        #     # Add to generated sequence
        #     generated_ids.append(next_token_id)
            
        #     # Stop if EOS token is generated
        #     if next_token_id == tokenizer.eos_token_id:
        #         break
    
    # DecKCn = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return captions

def main():
    st.title("📸 Image Captioning App")
    st.write("Upload an image or take a photo to get an AI-generated caption!")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable debug mode")
    
    # Load model (cached)
    with st.spinner("Loading model..."):
        model, processor, device = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Camera input option
    camera_image = st.camera_input("Or take a photo with your camera")
    
    # Process the uploaded image or camera image
    if uploaded_file is not None or camera_image is not None:
        input_file = uploaded_file if uploaded_file is not None else camera_image
        
        # Convert bytes to PIL Image
        image = Image.open(input_file).convert("RGB")
        
        # Display the image
        st.image(image, caption="Your Image", use_container_width=True)
        
        # Process image when user clicks the button
        if st.button("Generate Caption"):
            try:
                with st.spinner("Analyzing image..."):
                    # Show debug info if enabled
                    if debug_mode:
                        st.write("### Debug Information")
                        # st.write(f"Model text_seq_length: {model.text_seq_length}")
                        # st.write(f"Model image_seq_length: {model.image_seq_length}")
                        st.write(f"Device: {device}")
                    
                    # Time the caption generation
                    start_time = time.time()
                    caption = generate_caption(model, processor, image, device)
                    end_time = time.time()
                    
                    # Display the results
                    st.success(f"Caption generated in {end_time - start_time:.2f} seconds")
                    st.subheader("Generated Caption:")
                    st.write(f"### {caption}")
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")
                if debug_mode:
                    st.exception(e)
                
            # Add some information about the model
            with st.expander("About this model"):
                st.write("""
                This image captioning model uses a Vision Transformer (ViT) from CLIP 
                as the encoder, combined with a custom decoder architecture. The model 
                has been trained on the Flickr30k dataset, which contains 31,000 images 
                with 5 captions each.
                
                The model works by:
                1. Encoding the image using CLIP's vision model
                2. Using a masked self-attention mechanism to generate captions
                3. Generating tokens one-by-one in an autoregressive manner
                """)

if __name__ == "__main__":
    main()