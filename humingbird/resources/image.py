import transformers
from transformers import CLIPProcessor, CLIPModel
from PIL import Image as pil_image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

class Image:

    def predict(image_path=None, labels=None):

        if image_path == None:
            raise Exception("You did not specify an image path for a prediction! Use the `image_path` parameter to fix this error.")
        
        if labels == None or len(labels) == 0:
            raise Exception("You did not specify any labels for a prediction! Use the `labels` parameter to fix this error.")
        
        if len(labels) < 2:
            raise Exception("You must have at least 2 labels to make a prediction! Only 1 was found.")
        
        else:

            try:
                image = pil_image.open(image_path)

                inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                scores = probs[0].detach().numpy()
                return_object = []

                for count, value in enumerate(scores):
                    return_object.append({'className': labels[count], 'probability': round(float(scores[count]), 2)})
                
                return return_object
            
            except FileNotFoundError:
                raise Exception('No file found! Please make sure you are using an existing image file.')
