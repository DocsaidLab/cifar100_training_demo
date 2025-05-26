[中文](./README.md) | **[English](./README_en.md)**

# CIFAR-100 Training Example

## Sleepless Nights of Overfitting

This project provides a concise PyTorch training example using the CIFAR-100 image classification task, designed to help beginners quickly get started.

It also offers some flexible adjustment and extension sample code, making it easier for you to adapt to different experimental needs.

Without further ado, let’s get started.

## Download This Project

The initial preparation for the project is complete; please use the following command to obtain the code:

```bash
git clone https://github.com/DocsaidLab/cifar100-training-demo.git
```

## Setup Training Environment (Ubuntu 22.04/24.04)

> \[!IMPORTANT]
>
> **Why use Docker?**
>
> 1. **Consistency**: Ensures “If it runs on my machine, it will run on yours.”
> 2. **No Pollution**: All dependencies are packaged inside the image, avoiding messing up your existing Python/conda setup.
> 3. **Easy Reproducibility**: When errors occur, `docker rm` + `docker run` resets the environment instantly.
>    (If you are more familiar with venv/conda, feel free to set up your own environment; this project mainly uses Docker.)

### Build Docker Environment

This section is thoroughly explained in our foundational toolbox project, please refer to:

- [**Docsaid Capybara #Advanced Installation**](https://docsaid.org/docs/capybara/advance)

### Download and Build the Image

```bash
git clone https://github.com/DocsaidLab/cifar100-training-demo.git
cd cifar100-training-demo
bash docker/build.bash
```

- Base Image: `nvcr.io/nvidia/pytorch:25.03-py3`
- Version Details: [**PyTorch Release 25.03**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html#rel-25-03)
- First build download size \~20 GB, taking about 5–20 minutes depending on your network speed.

After completion, verify the image with:

```bash
docker images | grep cifar100_train
```

## Constructing the Dataset

In PyTorch, the CIFAR-100 dataset is already built into `torchvision`, so you can directly use:

```python
from torchvision.datasets import CIFAR100

train_dataset = CIFAR100(root='data/', train=True, download=True)
test_dataset = CIFAR100(root='data/', train=False, download=True)
```

But wait!

Since this is a practice exercise, why not try downloading and constructing the dataset yourself to better control the data processing pipeline?

First, download the CIFAR-100 dataset from the official website and extract it:

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar xvf cifar-100-python.tar.gz
```

After execution, you will see a folder named `cifar-100-python` in your working directory, containing training and testing data.

Its structure looks roughly like this:

```text
cifar-100-python/
├── train
├── test
├── meta
└── file.txt~
```

These are not image files but are packaged as Python pickle files. Therefore, when using them later, you need to read these files with the `pickle` module.

## Writing the Dataset Class

After obtaining the dataset, we need to write a PyTorch dataset class to load this data.

Here, we implement a simple `CIFAR100DatasetSimple` class:

```python
import pickle

import capybara as cb
import numpy as np

DIR = cb.get_curdir(__file__)

class CIFAR100DatasetSimple:

    def __init__(
        self,
        root: str = None,
        mode: str = 'train',
        image_size: int = 32,
        return_tensor: bool = False,
        image_aug_ratio: float = 0.5,
    ):

        if mode not in ['train', 'test']:
            raise ValueError("mode must be either 'train' or 'test'")

        if root is None:
            self.root = DIR / 'cifar-100-python'
        else:
            self.root = root

        self.image_size = image_size
        self.return_tensor = return_tensor

        # Load data files
        with open(f'{self.root}/{mode}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.images = data[b'data']
            self.labels = data[b'fine_labels']
            self.filenames = data[b'filenames']

        # reshape: (N, 3, 32, 32)
        self.images = self.images.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        img = cb.imresize(img, size=self.image_size)

        if self.return_tensor:
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img = img.astype(np.float32) / 255.  # simple normalization to [0, 1]
            label = np.array(label, dtype=np.int64)
            return img, label

        return img, label
```

This class offers several features:

1. Allows specifying the output image size (`image_size`), defaulting to 32.
2. Optionally converts images to PyTorch Tensor format (`return_tensor`).
3. Allows setting the dataset mode (`mode`), either training (`train`) or testing (`test`).

More advanced features can be added later; for now, let’s proceed to train the first baseline model.

## First Model: Acc = 44.26%

You can find some default config files inside the `config` folder. We will use these configs to control the training process.

For the first model, we use `resnet18_baseline.yaml`, employing the well-known ResNet-18 as the base model.

Before training, return to the parent directory:

```bash
cd ..
```

Then, start training with the following command:

```bash
bash cifar100-training-demo/docker/train.bash resnet18_baseline
```

Since this is the first model, let’s take a closer look at the parameter configuration.

### Key Configuration Explanation

In `config/resnet18_baseline.yaml`, the main settings are:

1. **Batch Size**: Set to 250, which divides evenly into the 50,000 training samples, simplifying training cycles.

2. **Image Size**: Set to 32, matching the original CIFAR-100 image size. Unless otherwise specified, this size will be used in subsequent experiments.

3. **Model Configuration**

   ```yaml
   model:
     name: CIFAR100ModelBaseline
     backbone:
       name: Backbone
       options:
         name: timm_resnet18
         pretrained: False
         features_only: True
     head:
       name: Baseline
       options:
         num_classes: 100
   ```

   - Uses `timm_resnet18` without pretrained weights (`pretrained: False`) to observe the model learning from scratch.
   - The `Baseline` head converts the backbone output into predictions for 100 classes.

4. **Training Epochs**: Set to 200. Through multiple trials, improvements beyond 200 epochs are minimal.

5. **Optimizer**: Uses `AdamW` with a learning rate (`lr`) of 0.001, providing relatively stable training performance.

6. **Weight Decay**: Set to 0.0001; small models naturally have some regularization, so this value can be lowered moderately.

---

Ultimately, this model achieved 44.26% test accuracy at epoch 186.

However, the training accuracy reached 100%, indicating classic overfitting.

## Adjusting Training Hyperparameters

Overfitting means the model memorizes the training data perfectly but fails to generalize to unseen data.

This is especially common on small datasets like CIFAR-100 because of many classes and few samples; the model tends to memorize details rather than learn general rules.

### Common Solutions Include

1. **Reduce Model Capacity**: Use smaller models to lower the risk of overfitting.
2. **Data Augmentation**: Random cropping, flipping, brightness adjustments, etc., to expose the model to more diverse images and enhance generalization.
3. **Regularization**: Techniques like Dropout and Weight Decay to constrain the model during training.
4. **Early Stopping**: Stop training early once validation accuracy plateaus to avoid overfitting.
5. **Use Pretrained Models**: If possible, fine-tune from models pretrained on large datasets (e.g., ImageNet) rather than training from scratch.
6. **Adjust Learning Rate and Batch Size**: Improper learning rates or batch sizes can destabilize training; tuning these can improve results.

---

We won’t discuss Early Stopping here; instead, we fix training at 200 epochs and report the highest accuracy achieved.

Data augmentation is a common technique, so let’s try it next.

## Data Augmentation: Acc = 36.48%

We attempt to improve model generalization by applying data augmentation.

Here, we introduce the `albumentations` library with some basic augmentations:

```python
import albumentations as A

class DefaultImageAug:

    def __init__(self, p=0.5):
        self.aug = A.OneOf([
            A.ShiftScaleRotate(),
            A.CoarseDropout(),
            A.ColorJitter(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=p)

    def __call__(self, img: np.ndarray):
        img = self.aug(image=img)['image']
        return img
```

The chosen augmentations include:

- **ShiftScaleRotate**: Random shifts, scaling, and rotations.
- **CoarseDropout**: Randomly masks parts of the image to simulate missing data.
- **ColorJitter**: Randomly adjusts brightness, contrast, and saturation.
- **HorizontalFlip**: Random horizontal flips.
- **VerticalFlip**: Random vertical flips.

Empirically, these augmentations often improve model generalization.

Next, we add this augmentation in `config/resnet18_augment.yaml`:

```yaml
dataset:
  train_options:
    name: CIFAR100AugDataset
    options:
      mode: train
      return_tensor: True
      image_aug_ratio: 1.0
  valid_options:
    name: CIFAR100AugDataset
    options:
      mode: test
      return_tensor: True
```

The result, however, was disappointing.

Test accuracy dropped to 36.48%, much lower than the previous 44.26%.

This happens because on CIFAR-100’s small 32×32 images, applying strong augmentations at once (e.g., ±45° rotation, large occlusion, or vertical flipping) severely distorts the original semantics, preventing the model from stably learning basic features.

## Strong Regularization: Acc = 40.12%

Next, we try to improve the model’s generalization by applying stronger regularization.

Generally, when training CNN models, the convolutional structure itself provides some translation invariance and parameter sharing, which acts as inherent regularization. Compared to Transformer models that tend to overfit easily in early training, CNNs usually do not require overly strong regularization.

Still, we give it a try.

Here, we increase the `weight_decay` to 0.1 to observe its effect on model learning and generalization.

In `config/resnet18_baseline_wd01.yaml`, the `weight_decay` setting is modified as:

```yaml
optimizer:
  name: AdamW
  options:
    lr: 0.001
    betas: [0.9, 0.999]
    weight_decay: 0.1
    amsgrad: False
```

As expected, the model’s test accuracy drops to 40.12%, lower than the original 44.26%.

This reflects a common phenomenon:

- For small datasets like CIFAR-100, applying overly strong regularization may suppress the model’s ability to fit the training data distribution sufficiently, causing premature convergence before learning sufficiently discriminative features, ultimately harming generalization.

## Label Smoothing: Acc = 44.81%

Next, we try using Label Smoothing to improve generalization.

Label Smoothing basically transforms the one-hot label vectors into smoothed distributions, which reduces overfitting to the training labels.

We configure this model using `config/resnet18_baseline_lbsmooth.yaml`.

The usage is straightforward, just add the `label_smoothing` parameter in the loss function:

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

The experiment shows the model reaches 44.81% test accuracy at epoch 59, which not only surpasses the previous 44.26% but also achieves this accuracy more than 100 epochs earlier.

This indicates Label Smoothing effectively reduces overfitting on this task and improves generalization.

## Ultimately, It’s Still a Data Problem

At this point, we can draw a realistic conclusion:

> **Some problems cannot be solved by model design or hyperparameter tuning alone.**

Take CIFAR-100 as an example: although it has a decent number of samples, the low resolution and sparse semantic information, combined with limited samples per class, make it hard for the model to learn robust discriminative features.

From a practical perspective, the most direct solution is: **more data**.

However, data collection is often costly.

In many real-world scenarios, data acquisition is difficult, and annotation is time-consuming and labor-intensive — a core bottleneck for deep learning deployment.

Thus, a more common and pragmatic choice is: **Transfer Learning**.

With transfer learning, we don’t train the model from scratch but leverage models pretrained on large datasets (e.g., ImageNet) as backbones, then fine-tune them on the target task.

This strategy offers several advantages:

- **Faster Convergence**: Initial weights already contain semantic features, so the model finds the learning direction faster.
- **Better Performance**: Even with limited target data, it fully exploits universal representations.
- **Reduced Overfitting**: Pretrained models provide a stable starting point, improving generalization.

Next, we will test using pretrained backbones provided by `timm`.

## Pretrained Weights: Acc = 56.70%

Continuing from the baseline setup, we temporarily do not use label smoothing or other regularization techniques, focusing on backbone pretrained weights.

We use `resnet18_pretrained.yaml` as the config, mainly adjusting the backbone part by setting `pretrained` to `True` to enable ImageNet pretrained weights.

```yaml
model:
  name: CIFAR100ModelBaseline
  backbone:
    name: Backbone
    options:
      name: timm_resnet18
      pretrained: True
      features_only: True
  head:
    name: Baseline
    options:
      num_classes: 100
```

At epoch 112, the model reaches 56.70% test accuracy, improving by **12.44%** compared to the original 44.26%.

This is a significant boost, much more effective than previous tuning tricks!

However, transfer learning is not a panacea. When the pretrained data domain and target task differ too much, the model may fail to transfer effectively or even suffer from **negative transfer**. For example, applying image pretrained models directly to natural language tasks rarely yields positive effects.

In our case, CIFAR-100 is a standard image classification task similar in context to ImageNet, so transfer learning naturally works well.

## Margin Loss: Acc = 57.92%

At this stage, we need to change our strategy.

If relying solely on the standard cross-entropy loss can no longer improve accuracy, we can try **actively increasing the training difficulty** to force the model to learn more discriminative feature representations. This is exactly what Margin Loss addresses.

### Why Margin?

In traditional classification, cross-entropy loss encourages the model to increase the logit score of the correct class, but **does not explicitly enforce a sufficient margin between the correct class score and incorrect ones**. In other words, as long as the correct class has the highest score, it’s acceptable—no matter by how much.

While this is enough for classification, when samples have close distributions, noisy data, or high inter-class similarity, the model’s decision boundaries can become ambiguous, leading to unstable generalization.

Margin Loss is designed to solve this problem:

> **Not only must the prediction be correct, it must be confidently correct.**

### What is Margin Loss?

The core idea of Margin Loss is:

> **To enlarge the distance between positive and negative samples in the logit or feature space, while reducing intra-class variance.**

Common margin losses include:

- **Large Margin Softmax (L-Softmax)**
- **ArcFace / CosFace / SphereFace**
- **Triplet Loss / Contrastive Loss**

These methods usually add an angular or magnitude margin before softmax, so the learned embeddings have clearer class boundaries in feature space. Below is a conceptual illustration of angular margin:

![margin_loss](./img/margin_loss.jpg)

The diagram shows Margin Loss pulling same-class features closer and pushing different classes farther apart, improving classification confidence and stability.

### Relation to Geometric Space

In practice, these losses often project features onto a unit hypersphere via L2 normalization, forcing embeddings to lie on a sphere with radius 1.

Benefits include:

- **Eliminating feature magnitude interference, focusing on direction (angle)**
- **Easier control of margin impact on angles**
- **Mathematically reformulating classification as an angular classification problem**

Hence, many margin-based methods apply margins on cosine similarity rather than directly manipulating logits.

### Experimental Results

Using pretrained ResNet-18 backbone, we incorporate Margin Loss in `config/resnet18_pretrained_arcface.yaml`.

We test two implementations, `ArcFace` and `CosFace`, with different margin settings:

```python
class ArcFace(nn.Module):

    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.theta = math.cos(math.pi - m)
        self.sinmm = math.sin(math.pi - m) * m
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class CosFace(nn.Module):

    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        logits[index, labels[index].view(-1)] -= self.m
        logits *= self.s
        return logits
```

After several experiments, we find the two methods perform similarly, with ArcFace slightly better.

Thus, we report the ArcFace result: at epoch 199, the model achieves 57.92% accuracy on the test set — a 1.22% improvement over standard Softmax loss.

This shows Margin Loss effectively enhances the model’s discriminative ability, especially when inter-class similarity is high, reducing overfitting and boosting generalization.

## Enlarged Input Images: Acc = 79.57%

Keeping the Margin Loss settings, we next try increasing the input image size to see if it further improves accuracy.

In `config/resnet18_pretrained_arcface_224x224.yaml`, we set `image_size` to 224:

```yaml
global_settings:
  image_size: [224, 224]
```

After enlarging the input images, the test accuracy peaks at 79.57% by epoch 29—an improvement of **21.65%** over the previous 57.92%.

This result is surprising:

- **Simply resizing the original $32 \times 32 $ images to $224 \times 224 $ greatly boosts model performance?**

There are several reasons:

1. **Resolution aligns with pretrained model expectations**

   ResNet-50 and most ImageNet models are pretrained on $224 \times 224 $ images. Feeding in $32 \times 32 $ images means convolutional kernels “see” almost the whole image at once, compressing hierarchical features and limiting detail discrimination. Enlarging input lets convolution layers extract textures and local structures at more reasonable receptive fields.

2. **Dramatic increase in spatial samples**

   From $32^2 $ to $224^2 $, pixel count increases **49 times**. Despite bilinear interpolation smoothing, the model captures more edges, textures, and color distributions, strengthening discriminative signals.

3. **Avoid early signal distortion and aliasing**

   At low resolutions, object details easily get averaged out by multiple stride/pooling layers; larger images preserve key features before downsampling. Higher resolution also reduces improper folding of high-frequency signals by convolution strides, maintaining feature stability.

---

Although accuracy significantly improves, some issues remain.

First, computation dramatically increases: training time jumps from \~3 minutes to \~2 hours (on RTX 4090).

Second, the model reaches \~80% accuracy within the first 30 epochs, meaning it learns most features early, with little gain afterward—indicating dataset information saturation, limiting further learning despite extended training.

## Larger Model Capacity: Acc = 61.76%

What if we keep input image size fixed but increase model capacity?

Generally, larger models can learn more features but risk overfitting.

Since Margin Loss reduces overfitting risk, we can try increasing model size.

Here, we use `resnet50_pretrained_arcface.yaml`, switching backbone to ResNet-50, while input size remains $32 \times 32 $:

```yaml
model:
  name: CIFAR100ModelMargin
  backbone:
    name: Backbone
    options:
      name: timm_resnet50
      pretrained: True
      features_only: True
  head:
    name: MarginHead
    options:
      hid_dim: 512
      num_classes: 100
```

Training results show test accuracy of 61.76% at epoch 199, an increase of 3.84% over 57.92%, at the cost of nearly doubling parameter count.

This shows that when input size can’t be increased, boosting model capacity still improves performance, especially combined with Margin Loss, which helps the model better learn class boundaries.

## Enlarged Input + Larger Model: Acc = 81.21%

Finally, we increase both model capacity and input image size to see if we can gain more.

In `config/resnet50_pretrained_arcface_224x224.yaml`, we set:

```yaml
global_settings:
  image_size: [224, 224]
```

With larger input, the model reaches over 80% accuracy within 5 epochs and peaks at 81.21% at epoch 174.

This result is close to ResNet-18 + 224x224’s, but with nearly double the parameters.

Clearly, the dataset has saturated, and increasing model capacity no longer yields significant gains.

## Knowledge Distillation: Acc = 57.37%

Maintaining ResNet-18 with $32 \times 32 $ inputs, to further improve performance we apply Knowledge Distillation (KD).

The core idea: transfer the discriminative ability learned by a large teacher model at high resolution to a lightweight student model.

Unlike traditional supervised learning relying only on hard labels, KD incorporates the teacher’s soft probability outputs as extra supervision. These soft labels encode inter-class relationships, guiding the student to learn a more discriminative feature space.

Distillation loss is defined as:

$$
\mathcal{L}_{\text{distill}} = (1 - \alpha)\,\mathcal{L}_{\text{CE}}(y, p_s) + \alpha T^2 \cdot \mathrm{KL}(p_t^{(T)} \,||\, p_s^{(T)})
$$

- $\mathcal{L} \_{\text{CE}} $: cross-entropy between student predictions and true labels.
- $\mathrm{KL} $: Kullback–Leibler divergence between teacher and student softmax outputs at temperature $T $.
- $\alpha $: balancing factor between true labels and distillation signal (commonly 0.5–0.9).
- $T $: temperature parameter to soften logits, emphasizing non-main-class differences.

---

In this experiment, a pretrained ResNet-50 trained on $224 \times 224 $ images serves as the teacher; the student is ResNet-18 with $32 \times 32 $ input.

The teacher model is frozen during training, only providing soft labels.

Training pipeline:

<div align="center">
    <img src="./img/teacher_student.jpg" width="60%">
</div>

1. Pretrain teacher model to obtain logits.
2. Apply temperature-scaled softmax to teacher and student logits to produce soft labels.
3. Train student with KD loss combining hard labels and distillation.
4. Deploy only the student model; no teacher needed at inference.

### Experimental Results

In `config/resnet18_pretrained_arcface_kd.yaml`, we set the parameters for knowledge distillation:

First, we load a pretrained ResNet-50 model based on 224 × 224 input size as the teacher:

```yaml
common:
  batch_size: 250
  image_size: [32, 32]
  is_restore: True
  restore_ind: "2025-05-26-00-49-22"
  restore_ckpt: "epoch=177-step=35600.ckpt"
  preview_batch: 1000
```

The results show similar performance to the original Margin Loss baseline, with test accuracy around 57.37%.

This indicates that the teacher model was not as helpful in this setting as we had expected.

Possible reasons include:

1. **Insufficient student model capacity**: ResNet-18’s representation space is much smaller than ResNet-50’s; distilling the fine-grained decision boundaries of a high-accuracy teacher may be too challenging for the student to mimic.

2. **Input resolution mismatch**: The teacher model was trained on 224 × 224 inputs, while the student uses only 32 × 32 images. This difference in resolution likely prevents the student from fully capturing the features learned by the teacher.

There may be other reasons, but these two are the primary hypotheses.

## Summary of Experiments and Results

Below is a table summarizing each experiment’s configuration and the final test accuracy on CIFAR-100:

| Config File                                | Description                                                                   | Accuracy |
| ------------------------------------------ | ----------------------------------------------------------------------------- | -------- |
| `resnet18_baseline.yaml`                   | ResNet-18, no pretraining, AdamW (lr=0.001), WD=0.0001                        | 44.26%   |
| `resnet18_augment.yaml`                    | Added Albumentations data augmentation (rotation, dropout, flips)             | 36.48%   |
| `resnet18_baseline_wd01.yaml`              | ResNet-18, no pretraining, Weight Decay set to 0.1                            | 40.12%   |
| `resnet18_baseline_lbsmooth.yaml`          | ResNet-18, no pretraining, Label Smoothing = 0.1                              | 44.81%   |
| `resnet18_pretrained.yaml`                 | ResNet-18, **ImageNet pretrained**                                            | 56.70%   |
| `resnet18_pretrained_arcface.yaml`         | ResNet-18 pretrained + Margin Loss (ArcFace)                                  | 57.92%   |
| `resnet18_pretrained_arcface_224x224.yaml` | ResNet-18 pretrained + Margin Loss, input image resized to 224×224            | 79.57%   |
| `resnet50_pretrained_arcface.yaml`         | ResNet-50 pretrained + Margin Loss, input remains 32×32                       | 61.76%   |
| `resnet50_pretrained_arcface_224x224.yaml` | ResNet-50 pretrained + Margin Loss, input image 224×224                       | 81.21%   |
| `resnet18_pretrained_arcface_kd.yaml`      | Knowledge Distillation (Teacher: ResNet-50 224×224; Student: ResNet-18 32×32) | 57.37%   |

## More to Explore

So far, we focused on ResNet-18 experiments with fixed input size of 32×32.

However, improving CIFAR-100 accuracy is not limited to these methods. In fact, on [Paper with Code’s leaderboard](https://paperswithcode.com/sota/image-classification-on-cifar-100), the best results have surpassed 96%.

Such top-performing models typically combine:

- Large ViT architectures or custom CNN designs
- High-resolution inputs
- Pretrained transfer learning and advanced data augmentations (e.g., RandAugment, MixUp, CutMix)
- Longer training schedules with Cosine Annealing or One-Cycle learning rate policies
- Modern regularization techniques like Label Smoothing and Sharpness-Aware Minimization
- Multi-model distillation and ensemble methods for final inference

These approaches may not suit all development scenarios, especially resource-constrained deployments, but they clearly show:

**Performance ceilings come not only from the model architecture but also from the holistic training strategy design.**

If you use CIFAR-100 as a training playground, don’t forget to try different architectures and strategy combinations, continuously validating and improving your results.
