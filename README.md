# Evaluation of Transfer Learning for Mask Detection

## 📌 Overview

With the rise of the COVID-19 pandemic, **mask compliance** in public spaces has become a critical concern. However, enforcing mask usage manually is neither scalable nor always effective. In this project, we explore the application of **Transfer Learning** with various deep learning architectures to automatically detect whether a person is wearing a mask.

We experiment with five popular CNN models:
- MobileNetV2  
- InceptionV3  
- ResNet50V2  
- VGG16  
- DenseNet121  

Using a custom-labeled dataset, we evaluate both **standard training/test splits** and **K-Fold Cross Validation** to assess the performance of these models.

---

## 🧠 Methodology

- Implemented using **Keras**, **TensorFlow**, and **Colab**.
- Applied **Transfer Learning** to reuse pre-trained models on our custom dataset.
- Used both:
  - `mask_detector.py` for standard training/testing
  - `k_fold_cross_validation.py` for 5-fold stratified cross-validation
- Employed **image augmentation** with `ImageDataGenerator` to reduce overfitting.

---

## 📈 Results

- **Standard Evaluation**:
  - All five models achieved near-perfect accuracy (~99%).
  
- **K-Fold Cross Validation**:
  - VGG16 performed best with an average accuracy of **98.6%**.
  - Highlights how standard evaluation can **overestimate model performance**.

---

## 🔧 Models Used

| Model         | Average Accuracy (CV) |
|---------------|------------------------|
| MobileNetV2   | ~97.4%                 |
| InceptionV3   | ~97.9%                 |
| ResNet50V2    | ~97.1%                 |
| VGG16         | **98.6%**              |
| DenseNet121   | ~97.5%                 |

---

## 🧪 Dataset

- Custom dataset located at:
  `/content/drive/MyDrive/Colab Notebooks/498R/Mask Detector/ALL_DATASETS/dataset`
- Data is organized in two classes: `with_mask`, `without_mask`
- Total of **3,846** images

---

## 📂 Project Structure

  * 📁 Mask-Detector/
  * ├── mask_detector.py # Transfer learning with selected model
  * ├── k_fold_cross_validation.py # K-Fold validation implementation
  * ├── training_labels.csv # CSV containing file-label mappings
  * ├── /ALL_DATASETS/dataset # Mask/No-Mask image folders
  * ├── /model_save/ # Saved models for each fold
  * ├── /logs/ # TensorBoard logs
  * ├── mask_detector.tflite # Exported TFLite model
  * └── mobilenetv2.h5 # Example saved model


---

## 🧪 Evaluation Metrics

Custom metrics implemented:
- **Precision**
- **Recall**
- **F1 Score**

Also included:
- Accuracy
- Loss
- Confusion Matrix (optional visualization)

---

## 🧰 Technologies Used

- **TensorFlow & Keras**
- **Google Colab**
- **OpenCV & Matplotlib**
- **ImageDataGenerator for Augmentation**
- **LabelBinarizer for Label Encoding**
- **TensorBoard for Training Visualization**

---

## ✅ How to Run

### 🖥️ Standard Training:

```bash
python mask_detector.py

## 🔁 K-Fold Cross Validation:

```bash
python k_fold_cross_validation.py
```

## 💾 Output:

* Saved .h5 model weights for each fold
* TensorBoard logs for training visualization
* Converted .tflite model for edge deployment

## 📊 Sample Results Visualization

* (Will add actual accuracy/loss plots below)
* Training/Validation Accuracy
* Training/Validation Loss

## 📝 Conclusion

** Transfer learning enables rapid deployment of mask detection systems.
** K-Fold Cross Validation is critical to understanding generalization.
** VGG16 stands out in robustness, making it a strong candidate for real-world deployment.

## 🙌 Acknowledgments

** Inspired by real-world pandemic enforcement challenges.
** Dataset compiled manually and preprocessed using standard pipelines.


---

**To Do Later**:

- Replace `images/accuracy_plot.png` and `images/loss_plot.png` with actual screenshots from your training results.
- Upload the model files (e.g., `.h5`, `.tflite`) and CSV.
- Create badges using GitHub Actions or Colab notebooks for quick launch if desired.

