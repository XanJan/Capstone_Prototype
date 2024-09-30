from diffusers import StableDiffusionPipeline
import gradio as gr

# Load the Stable Diffusion model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Move the model to the appropriate device
pipe = pipe.to(device)

# Define a function to generate the image from a text prompt
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Create the Gradio interface
# The first argument is a textbox for user input (the prompt), and the second is an image output
interface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="IKEA Living Room Generator",
    description="Generate images of living rooms with IKEA furniture based on a text description."
)

# Launch the interface
interface.launch()






