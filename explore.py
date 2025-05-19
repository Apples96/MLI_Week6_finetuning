from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM



# clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# print(clip)

lama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

print(lama)