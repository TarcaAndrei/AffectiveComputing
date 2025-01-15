import json
import torch
import time
from dataset_emotic import DatasetEmotic, CustomCollateFn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from transformers import pipeline
from PIL import Image


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

# dataset = DatasetEmotic(dataset_type="train")
# print(len(dataset))
# dataset_val = DatasetEmotic(dataset_type="val")
# print(len(dataset_val))
dataset = DatasetEmotic(dataset_type="test")
# print(len(dataset))
# print(dataset[1])
# print(len(dataset_test))



pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})


# Start the timer
start_time = time.time()

text_transcripts_for_images = {}
prompt_for_image = "USER: <image>\nPretend you are an expert in recognizing emotions of humans from just an image. For this particular image, try to understand the emotions of each human through the face experssion or the activity that it is doing and tell me those feelings as expressively as possible.\nASSISTANT:"
print("started the for")
# for i, el in enumerate(dataset):
for i in range(0, len(dataset)):
    imagine, _, _ = dataset[i]
    imagine = Image.fromarray(imagine)
    outputs = pipe(imagine, prompt=prompt_for_image, generate_kwargs={"max_new_tokens": 200})
    result_of_the_image = outputs[0]["generated_text"].split("\nASSISTANT:")[1].strip()
    text_transcripts_for_images[i] = result_of_the_image
    print(f"{i}....")
    # if i == 12000:
    #     break

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

# print(text_transcripts_for_images)

with open("captions_test.json", "w") as json_file:
    json.dump(text_transcripts_for_images, json_file, indent=4)
          