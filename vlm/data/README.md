# Dataset structure

Image preprocessing and training augmentation live in `vlm.data.image_transforms`.
Dataset implementations use its `preprocess_3d_image` and `get_training_image_transforms` functions.

## Datasets

New datasets should be implemented in `dataset.py`.  
All implemented datasets inherit from the `MedicalImageDataset`.  

### Data formatting
The formatting of the dataset happens in the `format_data` method.  
Here, first, the dataset is preprocessed by extracting the qa pairs in a dataset specific way. 
This is done using the dataset specific extractors defined in `extractors.py`.
See [here](#extractors) for details on the implementation of the extractors.


When instantiating a dataset, you neeed to specify which model you want to be using it for, as this determines how the dataset will be formatted.  
In the initialization of the dataset, this affects the dict that is passed to the model with the special tokens etc. and the implementation for the various models is in `format_into_message`. This returns 2 conversations: One with and one without the ground truth answer (`prompt_only`), for training and inference, respectively.

### Filtering the data
Data can be filtered by the answer types.  
Also, for debugging, `n_samples` can be specified to limit the dataset size.

### Fields to be (at least) expected after retrieving sample with __getitem__

Required based on extractors (See `extractors.py`, class `RequiredKeys`):
* `image_id`: The id of the image, e.g. a UID or the image file name
* `question`: The question that is posed. For muliple choice questions, this contains the answer options as well
* `answer`: The answer to the question
* `answer_type`: The type of answer, e.g. "CHOICE", "OPEN", "CLOSED"
* `system_text`: The system prompt

Added in `format_data`:
* `embedding_path`: Full path to the image embedding file
* `formatted_qa`: The formatted qa pair, including the answer to the question
* `formatted_qa_prompt_only`: The formatted qa pair, not including the answer to the question

Added in `Dataset` init (if `tokenize_fn` is passed -> else done in `collate_fn` of model):
* `tokenized_qa_text_tokens`: qa pair tokenized, including the answer to the question
* `tokenized_qa_text_prompt_only_tokens`: qa pair tokenized, not including the answer to the question

Added in `__getitem__`:
* `embedding`: Image embedding as np array
* `embedding_tensor`: Image embedding as torch tensor


## Extractors

The extractor class handles the sample extraction independently of the question formatting in the dataset classes.  
Each dataset has its own Extractor class. This usually implements a function like `extract_sample_list`, which iterates over all samples (typically loaded from a `dataset.json` in the dataset class) and brings them into an appropriate format, e.g. based on the type of question.  
Note that each sample should have some minimum required keys. These are defined in `RequiredKeys`. To make sure that each sample contains the required keys, the samples should be a `QAItemDict`, which checks at initialization that all required keys are present.

## Adding a new dataset

**1. Implement an extractor**  
As described [above](#extractors), implement an extractor that takes in a list of the raw sample dicts and transforms them in a way that they are of type `QAItemDict` with the [necessary keys](#fields-to-be-at-least-expected-after-retrieving-sample-with-getitem).

**2. Implement a dataset class**  
This class should inheret from `MedicalImageDataset`. In the `extract_sample_list` method, the extractor from step 1 will be used.
