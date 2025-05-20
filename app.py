"""
Streamlit app for image captioning with prompt guidance.
This app allows users to upload an image and provide a text prompt
to guide the caption generation process.
"""
import streamlit as st
import torch
from PIL import Image
import os
import time
from models import lora_image_captioning_model

# Set page config
st.set_page_config(
    page_title="Prompt-Guided Image Captioning",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the image captioning model"""
    # Set device - try to use MPS for Mac, fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        st.success("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        st.success("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        st.info("Using CPU")
    
    # Load model
    model = lora_image_captioning_model().to(device)
    
    # Load model weights
    model_path = "models/best_captioning_model.pt"
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if it's a state_dict or a checkpoint dictionary
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                # Display additional metadata if available
                training_info = {
                    "Trained epochs": checkpoint.get("epoch", "Unknown"),
                    "Validation loss": checkpoint.get("val_loss", "Unknown"),
                    "Training date": checkpoint.get("timestamp", "Unknown")
                }
                st.sidebar.write("### Model Training Info")
                for key, value in training_info.items():
                    st.sidebar.write(f"**{key}:** {value}")
            else:
                model.load_state_dict(checkpoint)
                
            st.success(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            st.error(f"⚠️ Error loading model: {str(e)}")
            st.exception(e)
    else:
        st.warning(f"⚠️ Model file not found at {model_path}")
        st.info("Running with untrained model weights for demonstration")
    
    # Set model to evaluation mode
    model.eval()
    
    return model, device

def generate_caption(model, image, prompt, device, max_tokens=50):
    """
    Generate a caption for the given image using the provided prompt.
    
    Args:
        model: The image captioning model
        image: PIL Image to caption
        prompt: Text prompt to guide the captioning
        device: Device to run inference on
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated caption string
    """
    # Move to device and time the generation
    start_time = time.time()
    
    # Generate caption
    with torch.no_grad():
        captions = model.generate(
            [image],  # Model expects a list of images
            [prompt],  # Model expects a list of prompts
            max_new_tokens=max_tokens
        )
    
    # Get the first caption from the batch
    caption = captions[0]
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    return caption, generation_time

def main():
    st.title("🖼️ Prompt-Guided Image Captioning")
    st.write("""
    Upload an image and provide a text prompt to guide the caption generation.
    Try prompts like "Describe this image:" or "What's happening in this scene?"
    """)
    
    # Debug mode toggle in sidebar
    debug_mode = st.sidebar.checkbox("Enable debug mode")
    
    # Sidebar options
    st.sidebar.title("Options")
    max_tokens = st.sidebar.slider("Maximum tokens to generate", 10, 100, 50)
    
    # Load model (cached)
    with st.spinner("Loading model..."):
        model, device = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Camera input option
    camera_image = st.camera_input("Or take a photo with your camera")
    
    # Prompt input
    prompt = st.text_input(
        "Enter a prompt to guide the caption generation",
        value="Describe this image in detail:",
        help="The prompt guides the model to generate a specific type of caption"
    )
    
    # Example prompts
    example_prompts = [
        "Describe this image in detail:",
        "What's happening in this scene?",
        "Write a creative story about this image:",
        "Explain what you see in this picture:",
        "List the main objects visible in this image:"
    ]
    
    selected_example = st.selectbox(
        "Or try one of these example prompts:",
        [""] + example_prompts
    )
    
    if selected_example:
        prompt = selected_example
        st.info(f"Using prompt: '{prompt}'")
    
    # Process the uploaded image or camera image
    if uploaded_file is not None or camera_image is not None:
        input_file = uploaded_file if uploaded_file is not None else camera_image
        
        # Convert bytes to PIL Image
        image = Image.open(input_file).convert("RGB")
        
        # Display the image
        st.image(image, caption="Your Image", use_column_width=True)
        
        # Process image when user clicks the button
        if st.button("Generate Caption", type="primary"):
            try:
                progress_text = "Analyzing image..."
                caption_placeholder = st.empty()
                caption_placeholder.info(progress_text)
                
                # Show debug info if enabled
                if debug_mode:
                    st.write("### Debug Information")
                    st.write(f"Device: {device}")
                    st.write(f"Prompt: '{prompt}'")
                    st.write(f"Max tokens: {max_tokens}")
                
                # Generate caption
                caption, generation_time = generate_caption(
                    model, image, prompt, device, max_tokens
                )
                
                # Display the results
                caption_placeholder.empty()
                st.success(f"Caption generated in {generation_time:.2f} seconds")
                
                # Display the caption with the prompt removed if found at the beginning
                display_caption = caption
                if prompt in display_caption:
                    # Remove the prompt from the beginning of the caption if present
                    display_caption = display_caption.replace(prompt, "", 1).strip()
                
                st.subheader("Generated Caption:")
                st.markdown(f"### {display_caption}")
                
                # Display raw output in debug mode
                if debug_mode:
                    st.write("### Raw Model Output:")
                    st.text(caption)
                
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")
                if debug_mode:
                    st.exception(e)
            
    # Add some information about the model
    with st.expander("About this model"):
        st.write("""
        This image captioning model combines a CLIP vision encoder with a Llama language model:
        
        - **Vision Encoder**: CLIP ViT-Base from OpenAI processes the image into visual features
        - **Text Decoder**: Llama 3.2 1B Instruct model generates text based on the image features
        - **Training**: The model was fine-tuned on the Flickr30k dataset using LoRA (Low-Rank Adaptation)
        
        The model works by:
        1. Encoding the image using CLIP's vision model
        2. Projecting image features to the text model's embedding space
        3. Conditioning the language model on both the image features and input prompt
        4. Generating a caption that follows the prompt and describes the image
        """)

if __name__ == "__main__":
    main()