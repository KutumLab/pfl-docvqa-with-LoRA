### PFL-DocVQA with LoRA
This repository contains the code for our submission to PFL-DocVQA Competition 2023 ([Track 2](https://benchmarks.elsa-ai.eu/?ch=2&com=introduction)) . 


Fine tuning can be achieved with either full finetuning or parameter efficient finetuning methods. Since we had to fine tune over the VisualTransformer 5 (VT5) model and also not make any changes in the architecture, we adopted parameter-efficient fine tuning with  Low-Rank Adaptation (LoRA) method. Additionally, we used LoRA because it significantly reduces the communication cost between client and server in a federated learning environment. To maintain differential privacy, we have added gaussian noise only to the LoRA parameters  but not to all trainable parameters. Since LoRA parameters are approximately 1% of all trainable parameters, the total noise added is also less to train the model. 



### Related Papers:
1. [Differentially Private Fine-tuning of Language Models](https://arxiv.org/abs/2110.06500)
2. [SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models](https://arxiv.org/pdf/2308.06522.pdf)

