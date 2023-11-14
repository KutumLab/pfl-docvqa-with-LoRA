# PFL-DocVQA with LoRA

This repositiory contains the code for our submission to [PFL-DocVQA Competition (Track 2)](http://158.109.8.94/?ch=2&com=introduction). Our contribution is applying LoRA to train the given VT5 model. By reducing the total no. of trainable parameters, LoRA reduces the total communication cost and total noise added to the model.

# Reference:

Related Papers:
1. [Differentially Private Fine-tuning of Language Models](https://arxiv.org/abs/2110.06500)
2. [SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models](https://arxiv.org/pdf/2308.06522.pdf)

