# 🧙‍♂️ Magical Invisibility Cloak

A Python project that creates a **real-time magical invisibility cloak effect** using your webcam.  
Inspired by **Harry Potter**, where the invisibility cloak makes the wearer disappear, this project lets you replicate the magic using computer vision! ✨

---

## 🌟 Inspiration
The idea for this project came from **Harry Potter's Invisibility Cloak**. I wanted to see if we could mimic that magical effect in real life using **Python, OpenCV, and NumPy**, and voilà — a real-time invisibility cloak on your webcam!

---

## 🛠️ Features
- Real-time cloak effect using your webcam
- Silky smooth edges with temporal smoothing and multi-scale Gaussian blur
- High refresh rate for smooth performance
- Customizable HSV color range to use different cloak colors

---
## 📦 Requirements
To run this project, you need **Python 3.x** installed on your system.

The project depends on the following Python packages:
- `opencv-python` (for webcam and image processing)
- `numpy` (for array operations)

You can install these dependencies easily using a `requirements.txt` file. Create a file named `requirements.txt` in your project folder with the following content:

opencv-python>=4.12.0
numpy>=1.25.0

Then, install the dependencies with:

```bash
pip install -r requirements.txt
```

## 🚀 How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/username/invisibility-cloak.git
cd invisibility-cloak
```
(Optional) Create a virtual environment and activate it:

bash:
python -m venv venv

### Activate Virtual Environment

**Linux / Mac:**
```bash
source venv/bin/activate
```
**Windows:**
```bash
venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
## Run the Cloak

```bash
python invisibility_cloak.py
```
## Instructions in the Webcam Window

- Adjust yourself in front of the camera  
- Press **ENTER** to capture the background  
- Wear a red cloak and watch the magic happen! ✨  
- Press **'q'** to quit


