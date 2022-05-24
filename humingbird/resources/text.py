import transformers
from transformers import pipeline

text_classifier = text_classifier = pipeline('zero-shot-classification', model='typeform/mobilebert-uncased-mnli')

class Text:
    
    def predict(text=None, labels=None):

        if text == None:
            raise Exception("You did not specify a text snippet for a prediction! Use the `text` parameter to fix this error.")
        
        if labels == None or len(labels) == 0:
            raise Exception("You did not specify any labels for a prediction! Use the `labels` parameter to fix this error.")
        
        if len(labels) < 2:
            raise Exception("You must have at least 2 labels to make a prediction! Only 1 was found.")
        
        else:
            predictions = text_classifier(text, labels)
            probs = predictions['scores']
            labels = predictions['labels']

            return_object = []

            for count, value in enumerate(probs):
                return_object.append({'className': labels[count], 'score': round(probs[count], 2)})
            
            return return_object 
