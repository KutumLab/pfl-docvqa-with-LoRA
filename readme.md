# PFL-DocVQA with LoRA

This repositiory contains the code for our submission to  [PFL-DocVQA Competition (Track 2)](http://158.109.8.94/?ch=2&com=introduction). Our contribution is applying LoRA to train the given VT5 model. By reducing the total no. of trainable parameters, LoRA reduces the total communication cost and noise added to the model. 

Related Papers:
1. [Differentially Private Fine-tuning of Language Models](https://arxiv.org/abs/2110.06500)
2. [SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models](https://arxiv.org/pdf/2308.06522.pdf)

## About the Competition: 
<div style="text-align: justify;">
The objective of the Privacy Preserving Federated Learning Document VQA (PFL-DocVQA) competition is to develop privacy-preserving solutions for fine-tuning multi-modal language models for document understanding on distributed data.
We seek efficient federated learning solutions for finetuning a pre-trained generic Document Visual Question Answering (DocVQA) model on a new domain, that of invoice processing.

Automatically managing the information of document workflows is a core aspect of business intelligence and process automation.
Reasoning over the information extracted from documents fuels subsequent decision-making processes that can directly affect humans, especially in sectors such as finance, legal or insurance.
At the same time, documents tend to contain private information, restricting access to them during training.
This common scenario requires training large-scale models over private and widely distributed data.

Please, if you plan to participate in the Competition, read the [participation instructions](https://benchmarks.elsa-ai.eu/?ch=2&com=tasks#participating_rules) carefully.
</div>

## How to use
To set up and use the framework please check [How to use](framework_documentation/how_to_use.md#how-to-use) instructions.


## Dataset

<div style="text-align: justify;">
The dataset is split into Blue and Red data. Moreover, the Blue training set is further divided into 10 different clients.
The whole dataset comprises around 1M question-answer pairs on 109,727 document images from 6,574 unique providers.
In this competition, we will only use a reasonable set of the full dataset. 251,810 question-answer pairs are available for training and validation, while 43,591 pairs will be used for testing.
The rest of the dataset will be available after the competition period.

If you want to download the dataset, you can do so in the [ELSA Benchmarks Competition platform](http://158.109.8.94/?ch=2&com=introduction).
For this framework, you will need to download the IMDBs (which contains processed QAs and OCR) and the images.
All the downloads must be performed through the RRC portal.
</div>

| Dataset    | Link                                        |
|------------|---------------------------------------------|
| PFL-DocVQA | [Link](https://benchmarks.elsa-ai.eu/?ch=2) |


## PFL-DocVQA models weights

<div style="text-align: justify;">
We provide pre-trained weights on SP-DocVQA dataset to allow the particiapnts start from a common starting point. 
</div>

| Model    |                                 Weights HF name                                  | Parameters |
|:---------|:------------------------------------------------------------------------------:|:----------:|
| VT5 base | [rubentito/vt5-base-spdocvqa](https://huggingface.co/rubentito/vt5-base-spdocvqa)  |   316M     | 

## Metrics

**Average Normalized Levenshtein Similarity (ANLS)** <br>
The standard metric for text-based VQA tasks (ST-VQA and DocVQA). It evaluates the method's reasoning capabilities while smoothly penalizes OCR recognition errors. <br>
Check [Scene Text Visual Question Answering](https://arxiv.org/abs/1905.13648) for more details.
