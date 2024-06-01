# Fine Tuning LLM Models

Introductory Guide where we will talk about Different Techniques of Fine Tuning LLM Models. Fine-tuning Large Language Models (LLMs) involves adapting a pre-trained language model to a specific task or domain by training it on a smaller, task-specific dataset. The main goal of fine-tuning is to leverage the general linguistic knowledge acquired by the model during its initial large-scale pre-training phase and refine it to perform better on a specific task or within a particular context.

<hr>

## Table of Contents

| Topics Covered | 

|------------------------| 

| [Quantization Intution](#quantization-intution)|

|[LORA Intuition](#lora-intution)|

|[QLORA  Intution](#qlora-intution)|

|[Finetuning with LLama2](#finetuning-with-llama2)|

|[1 Bit LLM Intution](#1-bit-llm-intution)|

|[Finetuning with Google Gemma Models](#finetuning-with-google-gemma-models)|

|[Contribution](#contribution)|

<hr>

# Quantization Intution

`Conversion from Higher Memory Format to lower memory format`

LLM weights are floating point (decimal) numbers. Just like it requires more space to represent a large integer (e.g. 1000) compared to a small integer (e.g. 1), it requires more space to represent a high-precision floating point number (e.g. 0.0001) compared to a low-precision floating number (e.g. 0.1). The process of quantizing a large language model involves reducing the precision with which weights are represented in order to reduce the resources required to use the model. GGML supports a number of different quantization strategies (e.g. 4-bit, 5-bit, and 8-bit quantization), each of which offers different trade-offs between efficiency and performance.

Reference Video - https://www.youtube.com/watch?v=ZKdMbQq5T30

<img title="" src="https://private-user-images.githubusercontent.com/50765800/258572660-ea2d2757-329a-415d-aefd-60fb89168cfd.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTcyNDY4MjgsIm5iZiI6MTcxNzI0NjUyOCwicGF0aCI6Ii81MDc2NTgwMC8yNTg1NzI2NjAtZWEyZDI3NTctMzI5YS00MTVkLWFlZmQtNjBmYjg5MTY4Y2ZkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjAxVDEyNTUyOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM5ZjIyODJiN2QzNzUzOWY3NWFmMjI3ZTg4YzI3ODUwN2UwNzNiNzhjZjdlMWQ4NmE4Zjg3ZDA5ZTA4MGYyOWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.INoB3Mvy_jTAvhQKxohqGv8Cx3x1BRQF3Viw3AvucbI" alt="">

`For example`

If we have a model with around 60 to 70 billion parameters, and their weights are stored in 32 bits, loading and working with that model is challenging. Most computers do not have enough RAM or GPU capacity to handle the extensive calculations with high precision. While we could use cloud resources like `AWS instances` to access sufficient RAM, this approach can be costly. 

A practical solution is to convert the high-precision weights to lower precision. Specifically, by converting the weights from 32 bits to 8 bits (`int8`), inference becomes easier, enabling us to work with the model or fine-tune it more effectively. This conversion process, known as `quantization`, speeds up weight processing and calculations. `Quantization` makes handling large models feasible by reducing memory requirements and increasing computational efficiency.

## How to Perform Quantization

First we need to understand how is data stored in FP 32 and FP 16. We are only looking at these 2 to get an general Idea. 

#### `Single Precision Floating Point 32`

`First Bit`  for sign or unsign

`Next 8 Bits` for Exponent

`Remaining 23 Bits` for Mantissa 

#### `Half Precision Floating Point 16`

`First Bit` for sign or unsign

`Next 5 Bits` for Exponent 

`Remaining 10 Bits` for Mantissa 

There are 2 Methods to Perform Quantization `Symmetric Quantization` and `Asymmetric Quantization`

### Symmetric Quantization

Symmetric quantization scales both positive and negative values by the same factor, with the zero point at the midpoint of the quantized range. Suitable for data that is symmetrically distributed around zero. Here we will be studying about Min-Max Scaler Technique

**Process**

- Determine the  $Xmax$  and $Xmin$ of the Data which needs to be quantized

- Determine the $Qmax$ and $Qmin$. Q here is the range of values of the desired memory format. If we want to convert to `uint8` then the ranges will be 255 and 0. We can also use $2^b - 1$ where `b` is the number of bits of desired format. For `uint8` we will have 256 - 1 =  255. 

- `Scale(Δ)` This is the Step size between quantized levels 
  
  $$
  Δ = \frac{Xmax-Xmin}{2^b-1}
  $$

- `Zero Point(Z)` Maps $Xmin$ to zero. Typically Zero for symmetric Quantization
  
  $$
  Z = 0

  $$

- Qunatize the FP value $x$ to $q$
  
  $$
  q = round(\frac{x}{Δ})
  $$

### Asymmetric Quantization

Asymmetric quantization scales positive and negative values differently, with a non-zero zero point. Better for skewed data where the distribution is not centered around zero. 

**Process**

- Calculate $Xmin$ and $Xmax$

- Determine $Qmax$ and $Qmin$ or $2^b-1$ where b is the number of bits of desired format

- `Scale(Δ)` The Step size betwwen quantized levels 
  
  $$
  Δ = \frac{Xmax-Xmin}{2^b-1}
  $$

- `Zero Point(Z)` Maps $Xmin$ to zero
  
  $$
  Z = round(-\frac{Xmin}{Δ}) 
  $$

- Quantized the FP value $x$ to q 
  
  $$
  q = round(\frac{x}{Δ}) + Z
  $$

This Process of Squezzing of values from higher format to lower format is known as `Calibration`. 

## Modes of Quantization

### Post Training Quantization

In this Technique the process of Calibration happens after the model is trained. We squeeze the processed and calculated weights in order to convert the pre-trained model into a quantized model. 

![WhatsApp Image 2024-05-30 at 20.24.02_3aac51c5](C:\Users\gj979\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\TempState\77C33D0FB152118E33778D34AE8A0473\WhatsApp%20Image%202024-05-30%20at%2020.24.02_3aac51c5.jpg)

##### `Quantization Steps`

* Instead of retraining the entire model, PTQ applies quantization techniques to the existing model after training.

* Quantization involves mapping the model’s weights and activations from their original floating-point representation to lower-precision data types (e.g., 8-bit integers).

* During PTQ, the following steps are typically performed:
  
  * **Quantizing Weights**: Convert the model’s weights (parameters) to lower precision (e.g., 8-bit integers or 16-bit floats).
  * **Quantizing Activations**: Quantize the intermediate activations produced during inference.
  * **Calibration**: Determine appropriate quantization parameters (scaling factors, zero points) based on a representative dataset.
  * **Conversion to Quantized Model**: Create a quantized model that uses integer-only data types for both weights and activations.
  * **Deployment**: Deploy the quantized model for inference.

`Problem`

This method of Calibration that we discussed above results in loss of accuracy and data.  The quantization process introduces rounding errors due to the limited precision of integer representations. To overcome this we have another method which is known as QAT or Quantization Aware Training. 

### Quantization Aware Training

Unlike post-training quantization, which quantizes an already trained model, QAT incorporates quantization constraints during the training process itself. The goal is to create a model that can be directly quantized (e.g., to 8-bit integers) without significant accuracy degradation.

##### `How does QAT Works`

During QAT, the model is trained with the knowledge that it will eventually be quantized. Quantization-aware layers and operations are introduced into the model architecture. These layers simulate the behavior of quantized operations during training. The model adapts its weights and activations to better align with the quantization constraints. `Calibration`, which determines quantization parameters (such as scaling factors), is performed during training using a representative dataset.

`Process`

* **Define the target hardware:** This involves specifying the type of hardware the model will eventually be deployed on (e.g., mobile device, embedded system). This helps determine the target precision level (e.g., how many bits) for weights and activations.

* **Model Design:** Design your neural network architecture as usual.

* **Quantization Calibration:** This step involves gathering statistics about the distribution of activations in the model during training. This information is crucial for defining the quantization parameters later. Techniques like power-of-two scaling or min-max quantization can be used here.

* **Insert Fake Quantization Operations:** Here, you modify the model's computational graph by inserting special operations that mimic the quantization process. These operations don't actually change the values during training, but they force the model to adapt to the lower precision it will encounter after quantization.

* **Quantization-aware Training Loop:** Train the model as usual, but with the inserted fake quantization operations. The model learns while considering the impact of quantization, making it more robust to the eventual precision reduction.

* **Post-training Quantization:** After training, the model undergoes actual quantization. The weights and activations are converted to the target precision level using the calibration data gathered earlier.

* **Fine-tuning (Optional):** In some cases, the quantized model might experience a slight accuracy drop. To compensate, a final fine-tuning step with the quantized model can be performed to recover some lost accuracy.

![1fd9fe08-9bbe-4b57-8b45-a58012a57288](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/1fd9fe08-9bbe-4b57-8b45-a58012a57288.png)

![dcc140ea-0923-4633-aa42-cdc96443ee8c](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/dcc140ea-0923-4633-aa42-cdc96443ee8c.png)

<hr>

# LORA Intuition

Lora stands for `Low-Rank Adaption` and is a widely used, parameter-efficient fine-tuning technique for training custom LLMs. 

###### `Background`

When adapting large pre-trained models, directly updating all parameters can be computationally expensive. LoRA addresses this by introducing low-rank matrices to approximate the changes needed during fine-tuning. 

**Full Parameter Fine Tuning Challenges**

- Update all Model Weights 

- Hardware Resource Constrainsts

#### `Concept`

LoRA decomposes the parameter updates into low-rank matrices. Instead of updating the full parameter matrix, **we update two smaller matrices whose product approximates the update**. `Instead of Updating all the weights, it will track changes`. It will track the changes of the new weights based on fine tuning. 

### Mathematical Formulation

Let $W \in \mathbb{R}^{m \times n}$ be a weight matrix in the pre-trained model. Instead of updating $W$ directly, LoRA approximates the update $\Delta W$ using two low-rank matrices $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$, where $r \ll \min(m, n)$.

The approximation is given by:

$$
\Delta W \approx AB
$$

#### Steps Involved

1. **Initialization:**
   
   * Initialize matrices A and B with random values or using some initialization technique.

2. **Forward Pass:**
   
   * During the forward pass, compute the output using the modified weight matrix: 
     
     $$
     W' = W + \alpha AB
     $$
     
      where α is a scaling factor.

3. **Backward Pass:**
   
   * During backpropagation, only update matrices $A$ and $B$, not the original weight matrix $W$.

![7246748f-341a-4bc6-ae0a-62581d55dcd3](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/7246748f-341a-4bc6-ae0a-62581d55dcd3.png)

### Benefits

* **Parameter Efficiency:** By using low-rank matrices, the number of trainable parameters is significantly reduced.
* **Faster Training:** Reducing the number of trainable parameters leads to faster training times.
* **Memory Efficiency:** Requires less memory compared to updating the full weight matrix.

### Applications

* **Fine-tuning large language models:** LoRA is particularly useful for adapting large models like GPT, BERT, and others.
* **Transfer Learning:** Efficiently transfer knowledge from a pre-trained model to a new task with minimal computational resources.

### Example

Suppose you have a weight matrix $W \in \mathbb{R}^{1024 \times 1024}$. Instead of updating all $1024^2$ parameters, you can choose $r=64$ and decompose the update into $A \in \mathbb{R}^{1024 \times 64}$ and $B \in \mathbb{R}^{64 \times 1024}$. This reduces the number of trainable parameters from $1,048,576$ to $131,072$.

![27ba6963-4ba3-4acd-863f-3348ad09bfa3](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/27ba6963-4ba3-4acd-863f-3348ad09bfa3.png)

<hr>

# QLoRA Intuition

QLoRA stands for `Quantized Low-Rank Adaptation` and is an advanced technique that combines quantization with low-rank adaptation to further enhance parameter efficiency and computational savings in training custom LLMs.

###### `Background`

While LoRA effectively reduces the number of trainable parameters by using low-rank matrices, QLoRA goes a step further by incorporating quantization. This technique addresses the computational and memory challenges even more efficiently.

**Challenges Addressed by QLoRA**

* Update all Model Weights with Reduced Precision
* Further Minimize Hardware Resource Constraints

#### `Concept`

QLoRA extends LoRA by applying quantization to the low-rank matrices. Instead of just decomposing the parameter updates into low-rank matrices, QLoRA quantizes these matrices to reduce the bit-width of the parameters, leading to significant memory and computational savings.

### Mathematical Formulation

Let $W \in \mathbb{R}^{m \times n}$ be a weight matrix in the pre-trained model. Instead of updating $W$ directly, QLoRA approximates the update $ΔW$ using two low-rank matrices $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$, where $r \ll \min(m, n)$. These matrices $A$ and $B$ are then `quantized`.

The approximation is given by:

$$
\Delta W \approx Quantize(A).Quantize(B)
$$

#### Steps Involved

1. **Initialization:**
   
   - Initialize matrices $A$ and $B$ with random values or using some initialization technique.

2. **Quantization:**
   
   - Quantize matrices $A$ and $B$ to lower precision (e.g., `8-bit` integers). This is typically done using a technique like min-max scaling or uniform quantization.

3. **Forward Pass:**
   
   - During the forward pass, compute the output using the modified weight matrix:
     
     $$
     W' = W + \alpha * (Quantize(A) * Quantize(B))
     $$
     
     where $α$ is a scaling factor.

4. **Backward Pass:**
   
   - During backpropagation, only update the quantized matrices $A$ and $B$, not the original weight matrix $W$.

### Benefits

* **Extreme Parameter Efficiency:** Combining low-rank adaptation with quantization dramatically reduces the number of trainable parameters and the memory footprint.
* **Even Faster Training:** Reduced precision in parameter updates speeds up the training process.
* **Enhanced Memory Efficiency:** Quantizing the low-rank matrices uses significantly less memory compared to both full-precision and low-rank-only updates.

### Applications

* **Fine-tuning large language models:** QLoRA is particularly useful for adapting extremely large models like GPT, BERT, and others, especially in resource-constrained environments.
* **Transfer Learning:** Efficiently transfer knowledge from a pre-trained model to a new task with minimal computational resources and memory requirements.

### Example

Suppose you have a weight matrix $W \in \mathbb{R}^{1024 \times 1024}$. Instead of updating all $1024^2$ parameters, you can choose $r=64$ and decompose the update into $A \in \mathbb{R}^{1024 \times 64}$ and $B \in \mathbb{R}^{64 \times 1024}$. Then, quantize these matrices to `8-bit` integers. This reduces the number of trainable parameters from `1,048,576` to `131,072`, and further reduces memory usage due to quantization.

<hr>

# Finetuning with LLama2

Using QLora to Finetune LLama 2 Model. The Jupyter file for this is available in this repo. Important Package and Libraries that we need are: 

`Transformers`  It’s a huggingface library for quickly accessing (downloading from hugging’s API) machine-learning models, for text, image, and audio. It also provides functions for training or fine-tuning models and sharing these models on the HuggingFace [model hub](https://huggingface.co/models). The library doesn’t have abstraction layers and modules for building neural networks from scratch like Pytorch or Tensorflow. Instead, it provides training and inference APIs that are optimized specifically for the models provided by the library.

`bitsandbytes` to perform Quantization. It is a lightweight wrapper around CUDA custom functions, specifically designed for 8-bit optimizers, matrix multiplication, and quantization. It provides functionality for optimizing and quantizing models, particularly for LLMs and transformers in general. It also offers features such as 8-bit Adam/AdamW, SGD momentum, LARS, LAMB, and more. I believe that the goal for bitsandbytes is to make LLMs more accessible by enabling efficient computation and memory usage through 8-bit operations. By leveraging 8-bit optimization and quantization techniques we improve the performance and efficiency of models.

`Parameter-Efficient Transfer Learning` aims to efficiently adapt large pre-trained models (such as language models) to downstream tasks with limited task-specific data.  It addresses the challenge of fine-tuning large models while minimizing the number of trainable parameters. PETL techniques include approaches like adapter modules, low-rank adaptation, and quantized low-rank adaptation. peft package is used for parameter efficient transfrer learning. It includes LoraConfig, PeftModel and many more things related to fine tuning and transfer learning

`trl`  stands for Transformer Reinforcement Learning and is a library that provides implementations of different algorithms in the various steps for training and fine-tuning an LLM. Including the Supervised Fine-tuning step (SFT), Reward Modeling step (RM), and the [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1909.08593) step. trl also has peft as a dependency so that you can for instance use an SFT Trainer with a PEFT method such as LoRA

`AutoModelForCausalLM` to load models from hugging face.The transformers library provides a set of classes called [Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto#auto-classes) that given the name/path of the pre-trained model, can infer the correct architecture and retrieve the relevant model. This [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) is a generic Auto Class for loading models for [causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling#inference).

`pipeline` for inference after we are done with fine-tuning. I think the pipeline is a great utility, there is a list of various pipeline tasks ([see here](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/pipelines#transformers.pipeline.task)) you can choose from like, “Image Classification”, “Text Summarization” etc. You can also select a model to use for the task, but you can also choose not to and the pipeline will use a default model for the task. You can add an argument that does some form of preprocessing like tokenization or feature extraction. 

```
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
```

`LoraConfig` From the peft library we import the [LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.LoraConfig) data class. LoraConfig is a configuration class to store configurations required to initialize the [LoraModel](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.LoraModel) which is an instance of a [PeftTuner](https://huggingface.co/docs/peft/main/en/package_reference/tuners#tuners). We’ll then pass this config to the SFTTrainer it will use the config to initialize the appropriate Tuner which again, in this case, is the LoraModel. 

```
# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
```

`PeftModel` Once we fine-tune a pre-trained transformer using one of the peft methods such as LoRA, we can save the LoRA adapter weights to disk as well as load them back into memory. PS: Adapters are basically the weights that PEFT modules fine-tune, these are separate from the base-model weights. <u><i><b>Using PeftModel we also have the option of loading the adapter weights into memory and then merging the base_model weights with the newly fine-tuned adapter wrights</b></i></u>

### Prompt Template in case of LLama 2

`System Prompt` to guide the model 

`User Prompt` to give instructions 

`Model Answer` is required

![](data:image/png;base64,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)

### Overview of Steps Required

#### Step A

- Define variables and parameters which will be responsible for loading the `llama-2-7b-chat-hf` chat model
- Define variables and parameters which will be responsible for training the model on the `mlabonne/guanaco-llama2-1k (1,000 samples)`, which will produce our fine-tuned model Llama-2-7b-chat-finetune

`QLoRA will use a rank of 64 with a scaling parameter of 16. We’ll load the Llama 2 model directly in 4-bit precision using the NF4 type and train it for one epoch`

#### Step B

- Load the dataset. Here our dataset is already preprocessed but, usually, this is where we would reformat the prompt, filter out bad text, combine multiple datasets, etc.

- Then, configure bitsandbytes for 4-bit quantization.

- Next, Load the Llama 2 model in 4-bit precision on a GPU with the corresponding tokenizer.

- Finally, Load configurations for QLoRA, regular training parameters, and pass everything to the SFTTrainer. The training can finally start!

#### Step C

* Use Text Generation Pipeline to Inference the fine tuned model
* Do not forget to match the input format with prompt format of LLama 2.

#### Step D

* Store the newly Fine Tuned LLama 2 model (LLama-2-7b-chat-finetune)
* Merge the weights from LoRA with the base model. Reload the base model in FP16 and use the peft library to merge everything

<hr>

# 1 Bit LLM Intution

<a href="https://arxiv.org/pdf/2402.17764">Paper Link</a>  1.58 Bits to be exact. In these every single Parameter (or weight) of the LLM is ternary `{-1, 0, 1}`. It Matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being significantly more cost-effective in terms of latency, memory, throughput, and energy consumption.  

![63685cbd-70c4-418f-acd9-07500a75a2ed](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/63685cbd-70c4-418f-acd9-07500a75a2ed.png)

##### `Problem`

In recent years, the field of AI has seen a rapid growth in the size and capabilities of Large LanguageModels (LLMs). These models have demonstrated remarkable performance in a wide range of natural language processing tasks, but their increasing size has posed challenges for deployment and raised concerns about their environmental and economic impact due to high energy consumption. One approach to address these challenges is to use post-training quantization to create low-bit modelsfor inference. This technique reduces the precision of weights and activations, significantly reducing the memory and computational requirements of LLMs. The trend has been to move from 16 bits to lower bits, such as 4-bit variants

##### `Solution`

Recent work on 1-bit model architectures, such as BitNet presents a promising directionfor reducing the cost of LLMs while maintaining their performance. Vanilla LLMs are in 16-bitfloating values (i.e., FP16 or BF16), and the bulk of any LLMs is matrix multiplication. Therefore,the major computation cost comes from the floating-point addition and multiplication operations. Incontrast, the matrix multiplication of BitNet only involves integer addition, which saves orders ofenergy cost for LLMs. As the fundamental limit to compute performance in many chips is power, theenergy savings can also be translated into faster computation 

In this work, we introduce a significant 1-bit LLM variant called BitNet b1.58, where every parameteris ternary, taking on values of {-1, 0, 1}. We have added an additional value of 0 to the original 1-bitBitNet, resulting in 1.58 bits in the binary system. BitNet b1.58 retains all the benefits of the original1-bit BitNet, including its new computation paradigm, which requires almost no multiplicationoperations for matrix multiplication and can be highly optimized. Additionally, it has the same energy consumption as the original 1-bit BitNet and is much more efficient in terms of memory consumption, throughput and latency compared to FP16 LLM baselines. Furthermore, BitNet b1.58 offers two additional advantages. Firstly, its modeling capability is stronger due to its explicit support for feature filtering, made possible by the inclusion of 0 in the model weights, which can significantly improvethe performance of 1-bit LLMs. Secondly, our experiments show that BitNet b1.58 can match fullprecision (i.e., FP16) baselines in terms of both perplexity and end-task performance, starting from a 3B size, when using the same configuration

### BitNet b1.58

BitNet b1.58 is based on the BitNet architecture, which is a Transformer that replaces `nn.Linear` with `BitLinear`. It is trained from scratch, with 1.58-bit weights and 8-bit activations. Compared to the original BitNet

<b>Quantization Function</b>:  To constrain the weights to -1, 0, or +1, we adopt an `abs mean quantization function`. It first scales the weight matrix by its average absolute value, and then round each value tothe nearest integer among {-1, 0, +1}: 

$$
\widetilde{W} = \text{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right),
$$

$$
\text{RoundClip}(x, a, b) = \max\left(a, \min(b, \text{round}(x))\right),
$$

$$
\gamma = \frac{1}{nm} \sum_{ij} |W_{ij}|.
$$

The quantization function for activations follows the same implementation in BitNet, except thatwe do not scale the activations before the non-linear functions to the range [0, Qb]. Instead, the activations are all scaled to [−Qb, Qb] per token to get rid of the zero-point quantization. This is more convenient and simple for both implementation and system-level optimization, while introduces negligible effects to the performance in our experiments. 

<b>`LLaMA-alike Components`</b> :  The architecture of LLaMA has been the defacto backbone for open-source LLMs. To embrace the open-source community, our designof BitNet b1.58 adopts the LLaMA-alike components. Specifically, it uses RMSNorm SwiGLU, rotary embedding and removes all biases. In this way, BitNet b1.58can be integrated into the popular open-source software (e.g., Huggingface, vLLM and llama.cpp2) with minimal efforts.

### Results

BitNet b1.58 starts to match full precision LLaMA LLM at 3B model size in terms of perplexity,while being 2.71 times faster and using 3.55 times less GPU memory. In particular, BitNet b1.58 witha 3.9B model size is 2.4 times faster, consumes 3.32 times less memory, but performs significantly better than LLaMA LLM 3B.

![c8e2d6cd-2dcd-4264-80ac-90c776756651](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/c8e2d6cd-2dcd-4264-80ac-90c776756651.png)

The performance gap between BitNet b1.58 and LLaMA LLM narrows as the model size increases. More importantly,BitNet b1.58 can match the performance of the full precision baseline starting from a 3B size. Similarto the observation of the perplexity, the end-task results reveal that BitNet b1.58 3.9B outperforms LLaMA LLM 3B with lower memory and latency cost. This demonstrates that BitNet b1.58 is aPareto improvement over the state-of-the-art LLM models.

![45805ef2-f7e7-4a6e-9322-a0d62b8c953c](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/45805ef2-f7e7-4a6e-9322-a0d62b8c953c.png)

<b>Memory and Latency</b>: In particular, BitNet b1.58 70B is 4.1 times faster than the LLaMA LLM baseline.This is because the time cost for nn.Linear grows with the model size. The memory consumptionfollows a similar trend, as the embedding remains full precision and its memory proportion is smallerfor larger models. Both latency and memory were measured with a 2-bit kernel, so there is still room for optimization to further reduce the cost.

![8ead1c27-527b-432e-85e4-3517706fc3f9](file:///C:/Users/gj979/OneDrive/Pictures/Typedown/8ead1c27-527b-432e-85e4-3517706fc3f9.png)

#### LLMs on Edge and Mobile

The use of 1.58-bit LLMs has the potential to greatly improve the performance of language models on edge and mobile devices. These devices are often limited by their memory and computational power, which can restrict the performance and the scale of LLMs. However, the reduced memory and energy consumption of 1.58-bit LLMs allows them to be deployed on these devices, enabling a wide range of applications that were previously not possible. This can greatly enhance the capabilities of edge and mobile devices and enable new and exciting applications of LLMs. Moreover, 1.58-bitLLMs are more friendly to CPU devices, which are the main processors used in edge and mobile devices. This means that BitNet b1.58 can be efficiently executed on these devices, further improving their performance and capabilities. 

<hr>

# Finetuning with Google Gemma Models

<a href="https://huggingface.co/google/gemma-2b">Gemma 2b Model</a>.  Finetuning Google Gemma Model. The Jupyter file for this is available in this repo. Steps that were used in [this](#finetuning-with-llama2) section are similary used here.  Important Libraries and Packages are: 

`Transformers` It’s a huggingface library for quickly accessing (downloading from hugging’s API) machine-learning models, for text, image, and audio. It also provides functions for training or fine-tuning models and sharing these models on the HuggingFace [model hub](https://huggingface.co/models). 

`bitsandbytes` to perform Quantization. It is a lightweight wrapper around CUDA custom functions, specifically designed for 8-bit optimizers, matrix multiplication, and quantization. It provides functionality for optimizing and quantizing models, particularly for LLMs and transformers in general. It also offers features such as 8-bit Adam/AdamW, SGD momentum, LARS, LAMB, and more. I

`Parameter-Efficient Transfer Learning` aims to efficiently adapt large pre-trained models (such as language models) to downstream tasks with limited task-specific data. It addresses the challenge of fine-tuning large models while minimizing the number of trainable parameters. PETL techniques include approaches like adapter modules, low-rank adaptation, and quantized low-rank adaptation. peft package is used for parameter efficient transfrer learning. It includes LoraConfig, PeftModel and many more things related to fine tuning and transfer learning

`trl` stands for Transformer Reinforcement Learning and is a library that provides implementations of different algorithms in the various steps for training and fine-tuning an LLM. Including the Supervised Fine-tuning step (SFT), Reward Modeling step (RM), and the [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1909.08593) step. trl also has peft as a dependency so that you can for instance use an SFT Trainer with a PEFT method such as LoRA

`AutoModelForCausalLM` to load models from hugging face.The transformers library provides a set of classes called [Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto#auto-classes) that given the name/path of the pre-trained model, can infer the correct architecture and retrieve the relevant model. This [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) is a generic Auto Class for loading models for [causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling#inference).

### HugginFace Token

* In order to access gemma model from huggingface we need the access token from huggingface.

* Crete your account if not already then create the access token in read format

* In Colab, add your key in secrets section with the name of HF_TOKEN or any other name

### Loading Gemma 2b and Qunatization Process

```
model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)
```

```
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])
```

### Finetuning the Gemma Model

`Dataset Used` [Abirate/english_qoutes](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FAbirate%2Fenglish_quotes)

```
lora_config = LoraConfig(
    r=64,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
```

`q_proj (Query Projection)`: This is a linear transformation applied to the query part of the attention mechanism.

It projects the query input into a different space to calculate attention scores.

`k_proj (Key Projection)`: Similar to q_proj, this is a linear transformation applied to the key part of the attention mechanism.

It helps in comparing keys with queries to compute attention scores.

`v_proj (Value Projection)`: This linear transformation is applied to the value part of the attention mechanism.

After attention scores are computed, they are used to weigh these values to get the output of the attention layer.

`gate_proj (Gate Projection)`: This is part of the feed-forward network within a transformer layer.

It usually involves a gating mechanism like GELU or sigmoid to control the flow of information.

`up_proj & down_proj (Up and Down Projections)`: These are linear transformations used in the feed-forward network of a transformer layer.

They typically increase (up_proj) or reduce (down_proj) the dimensionality of the input data as part of the processing.

##### Loading Dataset

```
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
```

##### Formatting the Prompt

```
def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}<eos>"
    return [text]
formatting_func(data["train"])
```

##### Finetuning using SFTTrainer

```
import transformers
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=20,
        # Copied from other hugging face tuning blog posts
        learning_rate=2e-4,
        fp16=True,
        # It makes training faster
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
    packing=False
)
trainer.train()
```

<hr>

# Contribution

Your contributions are highly valued, whether you are an experienced researcher or a novice in the field. Here are some ways you can contribute:

1. **Adding New Techniques:** If you have insights or have experimented with new fine-tuning techniques, consider adding them to the relevant sections. This will help broaden the scope of the repository.
2. **Providing Examples:** Practical examples and case studies are immensely helpful. Share your experiences, code snippets, or complete tutorials to illustrate how different techniques can be applied.
3. **Improving Existing Content:** Enhancing the clarity, accuracy, and depth of the existing material is always welcome. This includes correcting errors, refining explanations, and updating outdated information.
4. **Expanding Resources:** Contributions in the form of additional reading materials, useful tools, and related research papers can provide valuable context and support for users.

To ensure a smooth and efficient contribution process, please follow these guidelines:

1. **Fork the Repository:** Create a personal copy of the repository by forking it to your GitHub account.
2. **Create a New Branch:** Use a descriptive name for your branch that reflects the nature of your contribution.
3. **Implement Your Changes:** Make your additions or modifications in your branch. Ensure that your changes maintain the consistency and quality of the existing content.
4. **Submit a Pull Request:** Once you have completed your changes, submit a pull request. Include a detailed description of your contributions and any relevant context that can help in the review process.

I am deeply appreciative of your interest and effort in contributing to this repository. Together, we can create a valuable resource that supports and enhances the understanding of fine-tuning large language models for everyone in the community.

Thank you for your contributions and collaboration.
