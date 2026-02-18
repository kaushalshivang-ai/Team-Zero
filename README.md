# Team-Zero
<h1> <br> <b> Duality AI’s Offroad Semantic Scene Segmentation </b> </br> </h1>

<h2>Offroad Semantic Segmentation with DINOv2 </h2>
<h3>🎯 Project Aim </h3>
The goal of this project is to perform high-fidelity Semantic Segmentation in unstructured, offroad environments. By identifying terrain features like lush bushes, dry grass, rocks, and ground clutter, this model provides the "vision" necessary for autonomous navigation or environmental analysis in wild terrains where standard road-based models fail.

<h3>🚀 Key Features </h3>
Backbone: Leverages DINOv2 (ViT-S/14), a state-of-the-art self-supervised Vision Transformer, to extract robust, universal visual features.

Enhanced Multi-Scale Head: A custom segmentation head using depthwise separable convolutions and multi-scale fusion (3x3, 5x5, and 7x7 kernels) to capture both fine details and global context.

Combo Loss Architecture: Combines Cross-Entropy, Dice Loss, and Focal Loss to handle high class imbalance (e.g., small logs vs. large sky areas).

VRAM Optimized: Specifically designed for 4GB VRAM (RTX 3050) using Mixed Precision Training (AMP) and Gradient Accumulation.

Advanced Augmentation: Utilizes the Albumentations library for heavy spatial and color transformations to prevent overfitting.

<h3>🛠️ Working Process </h3>
Feature Extraction: The frozen DINOv2 backbone processes input images (266×476) and produces high-dimensional patch tokens.

Multi-Scale Fusion: The custom head projects these tokens into a 2D grid and processes them through parallel convolutional paths to identify objects of different sizes.

Optimization:

Mixed Precision: Speeds up training and reduces memory footprint by using float16 where possible.

Cosine Annealing: Smoothly decays the learning rate to find the global minimum.

Early Stopping: Monitors validation IoU to prevent wasted compute and overfitting.

Evaluation: Tracks per-class Intersection over Union (IoU) and Dice Scores to provide a granular view of model performance across all 11 terrain classes.

<h3>📂 Class Mapping </h3>
The model is trained to recognize 11 distinct offroad classes:
| Index | Class Name | Raw Pixel Value |
| :--- | :--- | :--- |
| 0 | Background | 0 |
| 1 | Trees | 100 |
| 2 | Lush Bushes | 200 |
| 3 | Dry Grass | 300 |
| ... | ... | ... |
| 10 | Sky | 10000 |

<h3>⚙️ Configuration & Setup </h3>
The script is controlled via a centralized CONFIG dictionary. Ensure your data paths are updated:

Prerequisites
Python 3.10+ (Recommended 3.11 or 3.12 for maximum stability)

PyTorch (CUDA 12.1+ for RTX 30-series)

Albumentations

Tqdm, Matplotlib, PIL

<h3>📊 Outputs </h3>
After training, the script generates:

best_model.pth: The weights with the highest Mean IoU.

training_curves.png: Visual tracking of Loss, mIoU, and Pixel Accuracy.

per_class_iou.png: A bar chart showing how well the model identifies specific obstacles like "Rocks" or "Logs."

training_report.txt: A detailed summary of the final and best metrics.

