#  Flower Classifier Project

This is a simple deep learning project that classifies flower images into 102 categories using a pre-trained model with TensorFlow and Keras.

---

##  Project Info

This project was developed as part of the **AI Programming with Python and TensorFlow Nanodegree from Udacity!**

You will first build and train an image classifier using a Jupyter notebook, then convert it into a Python script that can be run from the command line.

---

##  Model

- Model: VGG-based transfer learning
- Framework: TensorFlow / Keras
- Trained on: Oxford 102 Flower Dataset
- Output: Top predicted flower names and probabilities

---

##  Model Prediction Example

The model predicts the top flower class(es) with probabilities.

![Flower Prediction](assets/inference_example.png)

---

##  Download Model

If the trained model file (`flower_classifier.h5`) is too large for GitHub, download it from here:

👉 [Download from Google Drive](https://drive.google.com/file/d/15VJC4wrJQKOEaprd478sYGPfCYAMPegr/view?usp=drive_link)

---

## 📁 Files in the Project

```
p2_image_classifier/
├── predict.py                      # Command-line script
├── flower_classifier.h5            # Trained model (can be large)
├── label_map.json                  # Mapping of labels to flower names
├── Project_Image_Classifier_Project.ipynb  # Jupyter Notebook development
├── assets/                         # Example images
│   └── inference_example.png
├── test_images/                    # Sample flower images
├── requirements.txt                # Project dependencies
└── README.md                       # You're here
```

---

##  Notes

- You need a GPU to train the model efficiently.
- Dataset is not included in this repo (too large).
- This project is for educational purposes only.

---

