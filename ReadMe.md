This project implements an All-in-One Image Restoration model using a Transformer-based deep learning architecture. The model is designed to handle various image degradation tasks — such as denoising, deblurring, deraining, and dehazing — within a single unified framework.

🖼️ Overview
Traditional image restoration methods are often specialized for one specific degradation type. This project aims to build a generalized transformer-based model capable of restoring images under multiple degradation conditions, eliminating the need for separate models.

Key Features
Transformer backbone for efficient global context modeling
Supports multiple restoration tasks with a shared architecture
Input image size: 128 × 128 × 3 (RGB)
Can be extended to higher resolutions with minor modifications
Modular design for easy experimentation and fine-tuning
Model Architecture
Encoder–Decoder Transformer structure
Patch embedding to process 2D image data efficiently
Self-attention layers to capture long-range dependencies
Multi-task restoration heads for flexible degradation handling
