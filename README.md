# Fine Tuning LLM Models

Introductory Guide where we will talk about Different Techniques of Fine Tuning LLM Models. Fine-tuning Large Language Models (LLMs) involves adapting a pre-trained language model to a specific task or domain by training it on a smaller, task-specific dataset. The main goal of fine-tuning is to leverage the general linguistic knowledge acquired by the model during its initial large-scale pre-training phase and refine it to perform better on a specific task or within a particular context.

<hr>

## Table of Contents

| Topics Covered         | 
|------------------------| 
|[Quantization Intuition](#quantization-intuition)|
|[LoRA Intuition](#lora-intuition)|
|[QLoRA Intuition](#qlora-intuiition)|
|[Finetuning with LLama2](#finetuning-with-llama2)|
|[1 Bit LLM Intution](#1-bit-llm-intution)|
|[Finetuning with Google Gemma Models](#finetuning-with-google-gemma-models)|
|[Contribution](#contribution)|
<hr>

# Quantization Intuition

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

![WhatsApp Image 2024-05-30 at 20 24 02_aeef1100](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/ed1c5e44-3d78-4410-b2a7-f351decfe4f6)

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

![1fd9fe08-9bbe-4b57-8b45-a58012a57288](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/5d61940b-0270-460c-a6c4-ab1d13d7c43f)

![dcc140ea-0923-4633-aa42-cdc96443ee8c](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/0d7d8c45-30a4-4d03-acba-0f1d06b2c054)

<hr>

# LoRA Intuition

Lora stands for `Low-Rank Adaption` and is a widely used, parameter-efficient fine-tuning technique for training custom LLMs. 

##### `Background`

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

3. **Backward Pass:**
   
   * During backpropagation, only update matrices $A$ and $B$, not the original weight matrix $W$.

![7246748f-341a-4bc6-ae0a-62581d55dcd3](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/888f0922-b3ae-41ec-8f32-997b30bdf52d)

### Benefits

* **Parameter Efficiency:** By using low-rank matrices, the number of trainable parameters is significantly reduced.
* **Faster Training:** Reducing the number of trainable parameters leads to faster training times.
* **Memory Efficiency:** Requires less memory compared to updating the full weight matrix.

### Applications

* **Fine-tuning large language models:** LoRA is particularly useful for adapting large models like GPT, BERT, and others.
* **Transfer Learning:** Efficiently transfer knowledge from a pre-trained model to a new task with minimal computational resources.

### Example

Suppose you have a weight matrix $W \in \mathbb{R}^{1024 \times 1024}$. Instead of updating all $1024^2$ parameters, you can choose $r=64$ and decompose the update into $A \in \mathbb{R}^{1024 \times 64}$ and $B \in \mathbb{R}^{64 \times 1024}$. This reduces the number of trainable parameters from $1,048,576$ to $131,072$.

![27ba6963-4ba3-4acd-863f-3348ad09bfa3](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/cd73cd39-c088-4a3b-adfd-88ca8770c3cc)
<hr>

# QLoRA Intuition

QLoRA stands for `Quantized Low-Rank Adaptation` and is an advanced technique that combines quantization with low-rank adaptation to further enhance parameter efficiency and computational savings in training custom LLMs.

##### `Background`

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

![image](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/18c814bd-c1ad-45d7-aa8c-cb4247202cb9)

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

![63685cbd-70c4-418f-acd9-07500a75a2ed](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/e04b3ce8-2709-47ee-9e42-9dae1089f9e5)

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

![c8e2d6cd-2dcd-4264-80ac-90c776756651](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/e8bd8584-2864-4c31-8932-4ea95d861938)

The performance gap between BitNet b1.58 and LLaMA LLM narrows as the model size increases. More importantly,BitNet b1.58 can match the performance of the full precision baseline starting from a 3B size. Similarto the observation of the perplexity, the end-task results reveal that BitNet b1.58 3.9B outperforms LLaMA LLM 3B with lower memory and latency cost. This demonstrates that BitNet b1.58 is aPareto improvement over the state-of-the-art LLM models.

![45805ef2-f7e7-4a6e-9322-a0d62b8c953c](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/fdf983d1-644d-443a-a45c-f048539f2ba5)

<b>Memory and Latency</b>: In particular, BitNet b1.58 70B is 4.1 times faster than the LLaMA LLM baseline.This is because the time cost for nn.Linear grows with the model size. The memory consumptionfollows a similar trend, as the embedding remains full precision and its memory proportion is smallerfor larger models. Both latency and memory were measured with a 2-bit kernel, so there is still room for optimization to further reduce the cost.

![8ead1c27-527b-432e-85e4-3517706fc3f9](https://github.com/Gaurav-Van/Fine-Tuning-LLM-Models/assets/50765800/afedf82e-d84f-422b-9097-81af9e1ed177)

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
