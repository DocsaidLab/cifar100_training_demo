# CIFAR-100 資料集訓練範例

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
