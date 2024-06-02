import json
from transformers import pipeline

class ClassifierNode:
    def __init__(self):
        self.model = pipeline("zero-shot-classification")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"default": ""}),
                "candidate_labels_json": ("STRING", {"default": "{\"tags\": []}"}),
                "num_labels": ("INT", {"default": 10}),
                "include_scores": ("BOOLEAN", {"default": True}),
                "output_format": (["json", "text"], {"default": "json"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("classification_result",)
    FUNCTION = "classify"
    CATEGORY = "Classifier"

    def classify(self, text_input, candidate_labels_json, num_labels=10, include_scores=True, output_format="json"):
        try:
            # Parse candidate labels from JSON
            candidate_labels_data = json.loads(candidate_labels_json)
            labels = [item['tag'] for item in candidate_labels_data['tags']]

            # Perform zero-shot classification
            result = self.model(text_input, labels)

            # Prepare the output
            output = []
            for label, score in zip(result['labels'], result['scores']):
                if len(output) < num_labels:
                    if include_scores:
                        output.append({"tag": label, "score": score})
                    else:
                        output.append({"tag": label})

            if output_format == 'json':
                return (json.dumps({"tags": output}),)
            else:
                return (str({"tags": output}),)
        except Exception as e:
            return (json.dumps({"error": str(e)}),)