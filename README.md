LIVE DEPLOYMENT AT https://game-screenshot-classifier.onrender.com/

# Game Screenshot Classifier

> Classify any gameplay screenshot into one of 50 iconic video games using a deep learning model built from scratch.

---

## Live Demo

Upload a screenshot and get the predicted game title instantly!
<img width="1681" height="828" alt="upload_page" src="https://github.com/user-attachments/assets/ae50ea0a-4b75-411b-bd55-03121fa613cd" />


<img width="395" height="366" alt="Screenshot 2025-07-25 081609" src="https://github.com/user-attachments/assets/fce74a91-afe2-4cbf-9053-6e2345fb54e8" />

---

## Final Training Results

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

<img width="2205" height="944" alt="post_training_graph" src="https://github.com/user-attachments/assets/2df48f6b-f09a-4e09-9593-0bcfb337812a" />


---

## Model Architecture

<img width="976" height="555" alt="cnn" src="https://github.com/user-attachments/assets/4b75eb40-83d1-4250-ad0f-6c11927cc604" /><img width="923" height="771" alt="data_augment" src="https://github.com/user-attachments/assets/521bc432-a09c-4334-8950-7190d444fde4" />


> Built entirely from scratch using TensorFlow/Keras

* Input: 128×128×3 screenshots
* 3 × Conv2D blocks: 32 → 64 → 128 filters
* MaxPooling after each block
* Flatten → Dense(256) → Dropout(0.5)
* Output: 50-way Softmax
* Total Parameters: < 2 million

<img width="1016" height="276" alt="epoch_count" src="https://github.com/user-attachments/assets/1c2f9ef4-eb8e-4777-90cc-daca3775fceb" />
<img width="1000" height="328" alt="patience" src="https://github.com/user-attachments/assets/49cc6098-ea01-4123-b462-74c05692ce3a" />


---

## Dataset Overview

* **50 game titles**
* **500+ screenshots per class**
* Cleaned & resized to **128x128**
* Collected from Google Images (gameplay, HUDs, menus, cover art)

<img width="1402" height="381" alt="please_stop_torturing_me" src="https://github.com/user-attachments/assets/13117b73-383d-40a9-9dd8-4d58344c0786" />
<img width="442" height="128" alt="prediction" src="https://github.com/user-attachments/assets/83ed5104-d9ce-487d-bc7d-875923c48e2d" />


|   |

| - |

---

## How to Run Locally

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

## Flask Web App Workflow

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

## Why No Transfer Learning?

* **Educational goal**: Understand each layer in a CNN
* **Small dataset**: 25k images (128×128) easily trainable on laptop
* **Custom architecture**: tuned for screenshot visuals (logos, UI, HUDs)

---

## Training Details

| Feature        | Value                       |
| -------------- | --------------------------- |
| Loss Function  | Categorical Crossentropy    |
| Optimizer      | Adam (lr=5e-4)              |
| LR Scheduler   | ReduceLROnPlateau           |
| Early Stopping | Patience = 15               |
| Augmentations  | Rotation, Shift, Flip, Zoom |
| Dropout        | 0.5 before final Dense      |

<img width="2379" height="1180" alt="epoch_accuracy_graph" src="https://github.com/user-attachments/assets/8dbdcc37-3056-43ce-805e-51146e80b059" />


---

## Common Confusions

| Game A     | Game B         | Why?                              |
| ---------- | -------------- | --------------------------------- |
| Elden Ring | Dark Souls     | Similar setting & UI              |
| Valorant   | Counter-Strike | Tactical FPS, minimal HUD         |
| COD        | Battlefield    | Explosions, soldiers, military UI |

---

---

## Project Structure

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
├── /git_repo_pics/     # README visuals (graphs, UI)
```


---

## 🗣️💥 Results at a Glance

* ✅ **47% validation accuracy** on 50-way classification
* ✅ **55% training accuracy
* ✅ Trained from scratch, no transfer learning
* ✅ Clean local Flask app with image cleanup

**Built with TensorFlow, Flask , Pillow , NumPy , SciPy
