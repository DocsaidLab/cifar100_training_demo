# CIFAR-100 訓練範例

此專案提供一個簡潔的 PyTorch 訓練範例，使用 CIFAR-100 影像分類任務讓初學者可以快速上手。

同時也提供了一些彈性調整與擴充的範例程式碼，讓你能更輕鬆地因應不同實驗需求。

話不多說，這就開始吧。

## 下載本專案

專案前期準備已完成，請直接使用下列指令取得程式碼：

```bash
git clone https://github.com/DocsaidLab/cifar100-training-demo.git
```

## 建置訓練環境（Ubuntu 22.04/24.04）

> [!IMPORTANT]
>
> **為什麼使用 Docker？**
>
> 1. **一致性**：保證「我的電腦跑得動，你的也一樣」。
> 2. **免汙染**：所有相依套件封裝在映像檔內，不會把你原本的 Python/conda 弄亂。
> 3. **易重製**：出錯時 `docker rm` + `docker run` 整個環境瞬間歸零。
>    （若你對 venv/conda 更熟悉，亦可自行建環境；本專案將以 Docker 為主。）

### 建置 Docker 環境

這個章節在我們的基礎工具箱專案中有詳細說明，請參考：

- [**Docsaid Capybara #進階安裝**](https://docsaid.org/docs/capybara/advance)

### 下載並建置映像檔

```bash
git clone https://github.com/DocsaidLab/cifar100-training-demo.git
cd cifar100-training-demo
bash docker/build.bash
```

- 基底：`nvcr.io/nvidia/pytorch:25.03-py3`
- 版本詳細資訊：[**PyTorch Release 25.03**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html#rel-25-03)
- 首次建置下載量 \~20 GB，時間視網速約 5–20 分鐘。

完成後可確認映像檔：

```bash
docker images | grep cifar100_train
```

## 建構資料集

在 PyTorch 中，CIFAR-100 資料集已經內建於 `torchvision` 中，因此我們可以直接使用：

```python
from torchvision.datasets import CIFAR100

train_dataset = CIFAR100(root='data/', train=True, download=True)
test_dataset = CIFAR100(root='data/', train=False, download=True)
```

不過，等一下！

既然是要練習，我們不妨試看自己下載和建構資料集，這樣可以更好地控制資料處理流程。

首先從官方網站下載 CIFAR-100 資料集，並解壓縮：

```bash
wget https://www.cs.toronto.edu/\~kriz/cifar-100-python.tar.gz
tar xvf cifar-100-python.tar.gz
```

執行完之後，你可以在你的工作目錄中看到一個名為 `cifar-100-python` 的資料夾，裡面包含了訓練和測試資料。

其中的結構大概長這樣：

```text
cifar-100-python/
├── train
|-- test
├── meta
|-- file.txt~
```

這個不是影像檔，而是已經打包成 Python 的 pickle 檔案格式。因此，我們等一下在使用的時候，需要使用 `pickle` 模組來讀取這些資料。

## 撰寫資料集

有了資料集之後，我們需要來寫一份 PyTorch 的資料集類別來讀取這些資料。

我們簡單實作一個 `CIFAR100DatasetSimple` 類別：

```python
import pickle

import capybara as cb
import numpy as np

DIR = cb.get_curdir(__file__)

class CIFAR100DatasetSimple:

    def __init__(
        self,
        root: str=None,
        mode: str='train',
        image_size: int=32,
        return_tensor: bool=False,
        image_aug_ratio: float=0.5,
    ):

        if mode not in ['train', 'test']:
            raise ValueError("mode must be either 'train' or 'test'")

        if root is None:
            self.root = DIR / 'cifar-100-python'
        else:
            self.root = root

        self.image_size = image_size
        self.return_tensor = return_tensor

        # 讀取資料檔案
        with open(f'{self.root}/{mode}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.images = data[b'data']
            self.labels = data[b'fine_labels']
            self.filenames = data[b'filenames']

        # shape: (N, 3, 32, 32)
        self.images = self.images.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = np.transpose(img, (1, 2, 0)) # (C, H, W) -> (H, W, C)
        img = cb.imresize(img, size=self.image_size)

        if self.return_tensor:
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img = img.astype(np.float32) / 255.  # 直接簡單歸一化到 [0, 1]
            label = np.array(label, dtype=np.int64)
            return img, label

        return img, label
```

這個類別有幾個功能：

1. 可以指定輸出影像的大小（`image_size`），預設為 32。
2. 可以選擇是否將影像轉換為 PyTorch Tensor 格式（`return_tensor`）。
3. 可以指定資料集的模式（`mode`），分為訓練集（`train`）和測試集（`test`）。

更複雜的功能等等再說，現在讓我們先來 Train 第一個 Baseline 模型吧。

## 第一個模型：Acc=44.26%

你可以在 `config` 資料夾內找到一些預設的配置檔案，我們會透過這些配置檔案來控制訓練流程。

第一個檔案我們使用 `resnet18_baseline.yaml`，使用大家耳熟能詳的 ResNet-18 作為基礎模型。

訓練之前，我們先退回上層目錄：

```bash
cd ..
```

接著，我們可以使用以下指令來開始訓練：

```bash
bash cifar100-training-demo/docker/train.bash resnet18_baseline
```

既然是第一個模型，我們可以仔細看一下參數的配置。

### 關鍵配置說明

在 `config/resnet18_baseline.yaml` 中，主要配置如下：

1. **Batch Size**：設為 250，能整除 50000 筆訓練資料，簡化訓練週期。

2. **模型配置**

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

   - 使用 `timm_resnet18` 不帶預訓練權重 (pretrained=False)，方便了解模型從頭學習的過程。
   - `Baseline` 負責將 backbone 輸出轉換為 100 類別預測。

3. **訓練 Epoch 數**：設定為 200。經多次嘗試，超過 200 時改善幅度不明顯。
4. **優化器**： 採用 `AdamW`，學習率（`lr`）為 0.001，整體訓練表現相對穩定。
5. **Weight Decay**： 設為 0.0001；小型模型本身正則化不算差，可適度降低此值。

---

最終，這個模型在第 186 個 epoch 時，test-set 的準確率達到了 44.26%。

但 train-set 的準確率已經達到了 100％，這就是典型的過擬合現象。

## 調整訓練超參數

所謂的「過擬合」，就是模型把訓練資料背得滾瓜爛熟，卻沒辦法能套用在其他資料上。

在 CIFAR-100 這類小型資料集上，這種現象尤其常見，因為類別多、樣本少，模型很容易記住細節而不是學習規則。

### 常見的解法有幾個

1. **減少模型容量**：改用較小的模型，減少過度擬合的風險。
2. **資料增強（Data Augmentation）**：隨機裁切、翻轉、亮度調整，讓模型看更多圖，增加泛化能力。
3. **正則化（Regularization）**：使用 Dropout、Weight Decay 等手法，讓模型在學習時保持「克制」。
4. **提早停止（Early Stopping）**：當驗證集準確率不再上升時，提早結束訓練，避免過度擬合。
5. **使用預訓練模型（Pretrained Model）**：若允許，可以從大型資料集（如 ImageNet）微調過來，而非從頭訓練。
6. **學習率與 Batch Size 調整**：學習率太高或太低都會導致模型不穩，batch size 也會影響梯度更新穩定性。

---

Early Stopping 的部分我們就不討論了，反正固定跑 200 個 epoch，然後挑最高分數來報告。

資料增強是個常見的技巧，接著我們先來試試看。

## 資料增強：Acc=36.48%

我們來試試看使用資料增強的方式來改善模型的泛化能力。

這裡我們引入 `albumentations` 這個資料增強庫，加入一些基本的資料增強操作。

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

這邊選用的增強方法包括：

- **ShiftScaleRotate**：隨機平移、縮放和旋轉影像。
- **CoarseDropout**：隨機遮罩影像的一部分，模擬資料缺失。
- **ColorJitter**：隨機調整影像的亮度、對比度和飽和度。
- **HorizontalFlip**：隨機水平翻轉影像。
- **VerticalFlip**：隨機垂直翻轉影像。

經驗上，這些增強方法能有效提升模型的泛化能力。

接著，我們在 `config/resnet18_augment.yaml` 中加入這個增強方法：

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

結果卻讓人大失所望。

測試集的準確率只有 36.48%，遠低於之前的 44.26%。

這是因為在 CIFAR-100 這種解析度僅有 32×32 的小圖像中，若一次套用過多高強度增強（如旋轉 ±45°、大範圍遮蔽或垂直翻轉），會嚴重破壞圖像原始語意，模型無法穩定學習基本特徵。

## 強正則化：Acc=40.12%

接下來，我們嘗試使用正則化的方式改善模型的泛化能力。

一般來說，在訓練 CNN 模型時，由於卷積結構本身具備一定的平移不變性與參數共享特性，因此具有基本的正則化效果。相較於 Transformer 模型在訓練初期容易過擬合的特性，CNN 通常不需額外施加過強的正則化。

不過，我們還是可以試試看。

這裡將 `weight_decay` 提高至 0.1，觀察其對模型學習與泛化能力的影響。

在 `config/resnet18_baseline_wd01.yaml` 中修改 `weight_decay` 的設定：

```yaml
optimizer:
  name: AdamW
  options:
    lr: 0.001
    betas: [0.9, 0.999]
    weight_decay: 0.1
    amsgrad: False
```

實驗結果如預期所示：模型在測試集上的準確率下降至 40.12%，低於原始設定的 44.26%。

這反映出一個常見現象：

- 對於像 CIFAR-100 這類小型資料集而言，施加過強的正則化可能會壓抑模型在訓練階段對資料分布的充分擬合，導致模型尚未學會足夠區分性的特徵就過早收斂，最終影響泛化效果。

## Label Smoothing：Acc=44.81%

我們來試試看使用 Label Smoothing 的方式來改善模型的泛化能力。

Label Smoothing 的基本概念是將每個類別的標籤從 one-hot 編碼轉換為一個平滑的分布，這樣可以減少模型對訓練資料的過度擬合。

我們使用 `config/resnet18_baseline_lbsmooth.yaml` 來配置這個模型：

使用方式很簡單，只需要在損失函數中加入 `label_smoothing` 的參數即可。

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

實驗結果顯示，模型在第 59 個 epoch 時，test-set 的準確率達到了 44.81%，比之前的 44.26% 有所提升之外，也提早 100 多個 epoch 達到這個準確率。

這顯示出 Label Smoothing 在這個任務上能有效減少模型對訓練資料的過度擬合，並提升泛化能力。

## 終究還是資料不足

實驗進行到這裡，我們大致可以得到一個現實的結論：

> **有些問題，光靠模型設計或超參數微調是解決不了的。**

以 CIFAR-100 為例：雖然樣本數不少，但解析度低、語意訊息稀薄，每個類別的樣本數也有限。這樣的資料特性，使模型難以學得具泛化能力的判別特徵。

從實務角度看，最直接的解法就是：**增加資料**。

然而，資料蒐集往往是一項高成本工程。

在許多應用場景中，資料難以取得，標註又費時費力，這早已成為深度學習落地的核心瓶頸之一。

因此，實務界更常見也更務實的選擇是：**遷移學習（Transfer Learning）**。

透過遷移學習，我們不必從零開始訓練模型，而是善用在大規模資料集（如 ImageNet）上預訓練的模型作為 backbone，接著在目標任務上進行微調（fine-tune）。

這樣的策略具備多重優勢：

- **加速收斂**：初始權重已蘊含語意特徵，模型可更快找到學習方向
- **提升表現**：即使目標資料有限，亦能充分利用通用表示
- **降低過擬合**：預訓練模型為訓練提供穩定起點，泛化效果更佳

於是我們接下來就使用 `timm` 提供的預訓練模型來實際測試看看。

## 預訓練權重：Acc = 56.70%

這次我們使用 `resnet18_pretrained.yaml` 作為設定檔，主要調整的是 backbone 的部分，將 `pretrained` 選項設為 `True`，以啟用來自 ImageNet 的預訓練權重。

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

在第 112 個 epoch 時，模型於 test set 上達到 56.70% 的準確率，相比原本的 44.26%，提升了 **12.44%**。

可說是效果顯著，比剛才所有調餐的技巧都來得有用得多！

不過，遷移學習也非萬能。當預訓練資料與目標任務之間差異過大時，模型不僅可能無法有效遷移，甚至會產生所謂的「**負遷移（Negative Transfer）**」。例如，將影像預訓練模型應用於自然語言任務，幾乎無法發揮正面效益。

但在我們這個例子中，CIFAR-100 屬於標準的影像分類任務，與 ImageNet 的語境相近，遷移學習的效果自然也表現得相當理想。
