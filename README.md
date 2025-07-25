# 🎮 Game Screenshot Classifier

> Classify any gameplay screenshot into one of 50 iconic video games using a deep learning model built from scratch.

---

## ✨ Live Demo

Upload a screenshot and get the predicted game title instantly!

---

## 📈 Final Training Results

* **Validation Accuracy**: **47.2%** (50-way classification)
* **Top-3 Accuracy**: \~75%
* **Training Time**: \~2 hours (on CPU)
* **Model Size**: < 5 MB

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
| ----- | ----------------- | ------------------- | --------------- |
| 1     | 5.1%              | 19.6%               | 3.33            |
| 10    | 54.5%             | 41.7%               | 2.49            |
| 38    | 70%               | **52.4%**           | 2.01            |
| 50    | 74%               | **47.2%**           | 2.14            |

---

## 📊 Model Architecture

> Built entirely from scratch using TensorFlow/Keras

* Input: 128×128×3 screenshots
* 3 × Conv2D blocks: 32 → 64 → 128 filters
* MaxPooling after each block
* Flatten → Dense(256) → Dropout(0.5)
* Output: 50-way Softmax
* Total Parameters: < 2 million

---

## 📆 Dataset Overview

* **50 game titles**
* **500+ screenshots per class**
* Cleaned & resized to **128x128**
* Collected from Google Images (gameplay, HUDs, menus, cover art)

|   |
| - |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
https://github.com/yourusername/game_screenshot_classifier.git
cd game_screenshot_classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Flask app
python app.py

# 4. Open in browser
http://127.0.0.1:5000
```

---

## 🚪 Flask Web App Workflow

> Upload. Predict. Display.

```plaintext
upload.html → Flask → input/ → predict.py → display.html
```

* User uploads screenshot (upload.html)
* Image saved to `/input/`
* `predict.py` loads SavedModel + class\_indices
* Prediction is returned to user (display.html)
* Uploaded image is deleted immediately

---

## 🔮 Why No Transfer Learning?

* **Educational goal**: Understand each layer in a CNN
* **Small dataset**: 25k images (128×128) easily trainable on laptop
* **Custom architecture**: tuned for screenshot visuals (logos, UI, HUDs)

---

## 🔢 Training Details

| Feature        | Value                       |
| -------------- | --------------------------- |
| Loss Function  | Categorical Crossentropy    |
| Optimizer      | Adam (lr=5e-4)              |
| LR Scheduler   | ReduceLROnPlateau           |
| Early Stopping | Patience = 15               |
| Augmentations  | Rotation, Shift, Flip, Zoom |
| Dropout        | 0.5 before final Dense      |

---

## 💡 Common Confusions

| Game A     | Game B         | Why?                              |
| ---------- | -------------- | --------------------------------- |
| Elden Ring | Dark Souls     | Similar setting & UI              |
| Valorant   | Counter-Strike | Tactical FPS, minimal HUD         |
| COD        | Battlefield    | Explosions, soldiers, military UI |

---

## 🧳 Future Work

*

---

## 🌐 Project Structure

```plaintext
game_recognizer/
├── app.py                   # Flask backend
├── model.py                 # Model training script
├── predict.py               # Model inference script
├── /saved_model/            # SavedModel + class map
├── /dataset/                # 50 game screenshot folders
├── /input/                  # Temporary upload directory
├── /templates/              # Flask HTML views
│   ├── upload.html
│   └── display.html
├── /static/                 # Demo images, videos, icons
│   ├── demo_result.png
│   └── pred_valo.png, ...
├── /git_repo_pictures/     # README visuals (graphs, UI)
```

---

## 🚀 Results at a Glance

* ✅ **47% validation accuracy** on 50-way classification
* ✅ **75%+ top-3 accuracy** for real-world screenshots
* ✅ Trained from scratch, no transfer learning
* ✅ Clean local Flask app with image cleanup

---

## 🚀 Deploy This

> Want to show this off on Heroku, Render, or Replit?

* Clone repo
* `pip install -r requirements.txt`
* Push + deploy Flask app
* Keep `/saved_model/` in place
* You're done!

---

## 🚀 Star This Project If You Liked It!

Your feedback helps keep the stars glowing ⭐

---

**Built with TensorFlow, Flask, and lots of caffeine ☕**
