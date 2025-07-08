from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
import replicate
import os
import streamlit as st

script_dir = os.path.dirname(os.path.abspath(__file__))
imgdir = os.path.join(script_dir, "ad_assets/images")
os.makedirs(imgdir, exist_ok=True)

hf_token = st.secrets["HF_TOKEN"]
groq_key = st.secrets["GROQ_API_KEY"]
openrouter_key = st.secrets["OPENROUTER_API_KEY"]

images = []
scene_prompts = []


sdxl = InferenceClient(
    provider="hf-inference",
    api_key=hf_token
)

llama3 = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=groq_key
)


mixtral = ChatOpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mixtral-8x7b-instruct"
)


def generate_image_with_replicate(prompt, out_path):
    output = replicate.run(
        "stability-ai/sdxl:3b07b2ec8b75c1ca7a708aef62e7a26c1d07e14ae807c0b18b53e1b43c6dcc96",
        input={"prompt": prompt, "width": 1024, "height": 1024}
    )


# Prompts

# Storyboard
storyboard_prompt = ChatPromptTemplate.from_template(
    "You are an advertising creative director. Given this product or ad concept, write a storyboard for a video ad with a total length of 20 to 30 seconds. "
    "Describe 3 to 5 very concise scenes (1–2 short sentences each), each on a separate line. "
    "Each scene should be visually distinct, but clear and simple enough for quick comprehension. "
    "After each scene, write the estimated number of frames (at 7 fps) on a new line in the format: 'Frames: X'. "
    "Format:\n"
    "Scene 1: [Description] (X sec)\nFrames: Y\n"
    "Scene 2: ...\nFrames: ...\n"
    "Do not exceed 30 seconds total. Do not write dialogue. Only describe visuals and actions.\n"
    "Concept: {concept}"
)
storyboard_chain = storyboard_prompt | llama3

# SDXL prompt
sdxl_prompt_template = ChatPromptTemplate.from_template(
    "Write a concise, detailed prompt (max 200 characters) for Stable Diffusion XL to visualize this ad scene. "
    "Include only what should be in the image. Scene: {scene}"
)
sdxl_prompt_chain = sdxl_prompt_template | mixtral


st.title("Ad Storyboard & Image Generator")
st.write("Describe your ad concept below:")

ad_concept = st.text_area("Ad Concept", height=80)
if st.button("Generate Storyboard and Images") and ad_concept.strip():
    with st.spinner("Generating storyboard..."):
        storyboard_response = storyboard_chain.invoke({"concept": ad_concept})
        storyboard_text = (
            storyboard_response.content if hasattr(storyboard_response, "content") else storyboard_response
        )
    st.markdown("**RAW Storyboard Output:**")
    st.code(storyboard_text)

    # --- Parse scenes ---
    scenes = []
    frames_list = []
    lines = storyboard_text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.lower().startswith("scene"):
            desc = line.split(":", 1)[1].strip()
            scenes.append(desc)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.lower().startswith("frames:"):
                    frame_count = int(next_line.split(":")[1].strip())
                    frames_list.append(frame_count)
                    i += 1
        i += 1

    if not scenes or not frames_list:
        st.error("No scenes or frames extracted. Check storyboard output or parsing logic.")
    else:
        st.markdown("### Generated Scenes and Images")
        for idx, (desc, fnum) in enumerate(zip(scenes, frames_list), 1):
            st.markdown(f"**Scene {idx}:** {desc} ({fnum} frames)")
            with st.spinner("Generating SDXL prompt and image..."):
                sdxl_prompt = sdxl_prompt_chain.invoke({"scene": desc}).content
                st.caption(f"SDXL Prompt: {sdxl_prompt}")
                try:
                    # Generate image with Hugging Face InferenceClient
                    img = sdxl.text_to_image(
                        sdxl_prompt,
                        model="stabilityai/stable-diffusion-xl-base-1.0",
                    )
                    # Save and show image
                    img_filename = os.path.join(imgdir, f"scene_{idx}.png")
                    img.save(img_filename)
                    st.image(img, caption=f"Scene {idx}", use_column_width=True)
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
else:
    st.info("Enter your ad concept and click 'Generate Storyboard and Images'.")