# CNN 影像分類器之迭代與優化  
**CNN Image Classifier: Iteration & Optimization**

**學生：** 許家羚  
**學號：** 114153016  

---

## 專案主題
本專案由課堂中老師提供的**單通道灰階人臉影像分類模型**出發，透過自主更換資料庫與修改程式碼，將模型擴展為可處理**三通道 RGB 多類別彩色影像（貓咪）**，以驗證模型對不同資料特性與輸入維度的適應能力。
資料庫：https://drive.google.com/drive/folders/1cdeGu41ZecdKT7xNV_XXen6Fw5x_-b4O?usp=share_link



---


## 核心迭代重點

### 1. 資料特性調整
- **原始版本：** 單通道（Grayscale）人臉影像  
- **迭代版本：** 三通道（RGB）貓咪多類別影像  
- **目的：** 測試模型是否能正確調整輸入通道數（Cin）並正常運作  

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=3):
        super(SimpleCNN, self).__init__()
````

---

### 2. 模型彈性優化

* **原始版本：** 僅保留最後一輪訓練權重
* **迭代版本：** 自動儲存最佳準確率（Best Accuracy）模型
* **目的：** 防止過擬合（Overfitting），提升模型泛化能力

```python
if current_acc > best_acc:
    best_acc = current_acc
    torch.save(model.state_dict(), 'best_model.pth')
```

---

### 3. 視覺化診斷

* **原始版本：** 單圖輪播、純數字標籤
* **迭代版本：**

  * Loss / Accuracy 雙圖並列
  * 自動標註 Best Epoch
  * 網格化推論結果與錯誤色彩回饋
* **目的：** 提升模型收斂判斷與錯誤分析效率

---

## 心得與結論

在老師課堂講解的基礎上，我嘗試自主更換資料庫並修改程式碼完成模型迭代。過程中雖透過生成式 AI 協助理解與實作，但在確認程式能正常執行後，仍持續主動閱讀與解析程式碼細節，以加深對模型架構與運作原理的理解。

