## Humingbird
Humingbird is a Python framework for highly simplified machine learning classification. Using Humingbird allows you to build custom image and text classification systems with no data, no training and only 9 lines of code. Humingbird is intended for two types of developers:

- Developers who are experienced in programming but not in machine learning
- Developers experienced with machine learning but an extremely streamlined experience with little overhead.

**Note:** This package requires download of two models (one for image and one for text classification tasks). These models total approx 600 MB in storage.

## Quickstart
First, we need to download the package. You can do this by running the following command in your terminal:

```
pip install humingbird
```

After installation, we can run a custom image classification job with **no data or training!**

```python
import humingbird

# flexible creation + prediction in one call
prediction = humingbird.Image.predict(
       image_file="your_path.jpg",             # path to the image file for inference
       labels=["cat", "dog"]                   # supply potential labels that this image could be (i.e: allow the model to select the most probable)
)

print(prediction)
```

Which will yield something along the lines of:

```
[
    {
       "label": "cat",
       "score": 0.98
    },
    {
       "label": "dog",
       "score": 0.02
    }
]
```

It's as simple as providing the possible labels and a predicting image/text snippet.

## Using Humingbird for Text classification
Much like the quickstart, Humingbird also allows for highly simplified text classification workflows. 

```python
import humingbird

# flexible creation + prediction in one call
prediction = humingbird.Image.predict(
       text="I love Humingbird! It's so easy.",        # inference text snippet
       labels=["toxic", "not toxic"]                   # supply potential labels that this text snippet could be (i.e: allow the model to select the most probable)
)

print(prediction)
```

Which will yield the same type of output as the quickstart.

## How is this possible? 
Recent advances in the Transformer architecture has allowed for unprecendented capabilities in machine learning. One of these areas that has been recently improved in the area of **zero-shot-learning**.

A very basic way to think about zero-shot-learning is we train a model on lots of data for the task of predicting the most probable set of labels given an input. This may sound exactly like traditional classification, but where is differs is that we are finding the relationships between label -> input, similar to an image captioning problem.

After training these models in this fashion, what we get is a model that can do zero-shot-learning relatively well. This opens up the possibility to get rid of long training cycles, big data collection and complex setup for basic tasks.

## Limitations
This package can only do image and text classification. If I can get to it, I would love to support other modes of zero-shot-classification like object detection, entity recognition etc.

## Humingbird API
For those interested, a REST API for enhanced models may be on the roadmap. This would be for developers who don't have the hardware to run the models in the open source package, or would like a fully managed experience with premium features.


For more info, please check out https://www.humingbird.co
