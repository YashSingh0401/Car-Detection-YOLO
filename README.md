# 🚗 YOLO Car Detection Web App

A real-time **Object Detection Web App** built using **YOLOv8** and **Streamlit**.
This project detects vehicles (cars, buses, trucks, etc.) from images using deep learning.

---

## 🚀 Features

* 🔍 Real-time object detection using YOLOv8
* 📦 Detects multiple objects in a single image
* 🖼️ Upload image and get instant results
* ⚡ Fast and lightweight model (yolov8n)
* 🌐 Interactive web interface using Streamlit

---

## 🧠 Tech Stack

* Python 🐍
* YOLOv8 (Ultralytics)
* Streamlit
* OpenCV
* NumPy

---

## 🖼️ Demo

![Image](https://images.openai.com/static-rsc-4/016FFSYf9xviED81PTGtDqR_Cbad6qXfU_nOBfwqP0gySGFJ84vJuTNMi4VgVMzJT4VypBFt2jkHOgxEfvtATQ77wQBvQ7uY9NCcBINr2ZfqQG7mB6JjsUbv0AFhKj_RqvZxnIIXEJyioeJ8F6E-HUKY6hO_qzoJp9b8D2wddMoTTfO5dibHhvzByhJhPSD2?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/ysIXsDLR5jOD0wide84d0_0JqdXbz5fwQi2Ly8_Bc2ybii0Rtm4gRHr5__gUgosuAEsAIGUcrksa2abkynayN7D_uP_Epi5bC2dGN03zJ2c3xuqEuDR3mrpr_BUAVn31FtFpM6V-yTpS3b2bh4rNOhUKl3LDSCVznFf2sA0B3sxYOY3LRaDeyf59vQHD6QE7?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/NKsYTnwbhn7RRwNN09NUpXv4Qi0TPG9ZPCOl0CpTSp9EX8UA1PlyubPDQBPESArNR4ZLPC2Gazt33XVndrY9mZ11hOB-8L_48GwsbtrrU9MsExemFQVZS2Z9SWHLkbjzh4G681Vu4Ebua_BCPggayG25TlMITBDvgTcPbWWvN6_SZenCkQoAJ_BPJJLfhDqM?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/uw6Zmp0AotX_s4ES59yvaS-nzvb4YFrm73UPp0U10Jd_7WTX_kSIMX1fnKWqt7mEhpjAx8GdgTzynEXSr12dVd7B4vRkkPTgBm9JItRd-79QWLDlZ92-vYODKDxYohQGnXv5dnJwDSHthpIM1rvr45TnJxx1MzTd7NuvA4rCKK5ne6fdLGQoCONuXP7A_9Zi?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/WV62kvPHBLqAl1aLdfS8oZvDNcUAHggmDSKNV4r2aDPXr7LJY9Bjk8nMmRHWB2GOGcuxSPBVzP0GomVyYOs2YcDm5N-LcVhk6s-t_ZSGrxuODj6aLa6gohct0Q5X6wWoMrLVq31kSY1Agac9FrSMxyTAX2YWqLHIc3vXrWZ3OaPb7DgSNrgOY6WkDDs1UwfF?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/LXVIB6OJ4qnF3cx1Ki2Duq0JfkpeplEXH3TTtCT065MdwO5AhmAzW5djn5IxMULOTJF_A-DiyFIrnB80OS1CM0InxSLVhuaegzPN3tUfFWBiL5zvMr9PvrmPG6Jm-kSTysLWiXXxgG4tfMtCyDahCtHODWZ-HiBnx4Yghp9-4K7sNbRv9LsKwPAcRBvxNBXX?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/3FfdARZB6sl4n8NYkEXvT72fz0lYK03QZy1ViQk3BWcwCA8YUh_oYtPzlRUDJ7uV4keGOxrw0-z24OrwXTwpYD9nQ1T0eu61tx0kiXRgE25UYSUig7myPszDQwE6EUgO9yjtT-na2XBtlHxdGFWMxtJv6055IJda84zI1YRrd17yBJuxIW4174oepRipzb6s?purpose=fullsize)

---

## 📁 Project Structure

```
yolo-app/
│── app.py
│── yolov8n.pt
│── bus.jpg
│── requirements.txt
│── venv/
│── runs/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/yolo-car-detection.git
cd yolo-car-detection
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

**Windows:**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📸 How It Works

1. Upload an image
2. YOLO model processes the image
3. Detects objects with bounding boxes
4. Displays results instantly

---

## 📊 Model Used

* **YOLOv8 Nano (yolov8n.pt)**
* Pre-trained on COCO dataset
* Optimized for speed and efficiency

---

## 🌍 Deployment

You can deploy this app on:

* Hugging Face Spaces
* Streamlit Cloud

---

## 🔮 Future Improvements

* 🎥 Real-time webcam detection
* 🎯 Custom trained dataset
* 📱 Mobile-friendly UI
* 📊 Detection statistics dashboard

---

## 🙌 Author

** Yashwardhan Singh Sengar **
B.Tech AI/ML Student

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
