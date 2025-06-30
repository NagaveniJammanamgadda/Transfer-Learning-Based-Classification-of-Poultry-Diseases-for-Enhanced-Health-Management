# Transfer Learning-Based Classification of Poultry Diseases for Enhanced Health Management

**Project by:** [SmartInternz]

## Project Description

This project aims to develop a transfer learning-based system for classifying poultry diseases into four categories: **Salmonella, New Castle Disease, Coccidiosis, and Healthy**. The solution leverages a robust machine learning model (EfficientNetB3) integrated into a web application. Farmers and users can upload poultry images to receive an immediate diagnosis and suggested treatments. The goal is to empower farmers and stakeholders to manage poultry health more effectively, reducing disease impact and improving productivity.

### Key Features

- Deep learning model using transfer learning (EfficientNetB3)
- Web interface for image upload and disease prediction
- Four disease classes: Salmonella, New Castle Disease, Coccidiosis, Healthy
- Fast, user-friendly, and accessible

## Real-World Scenarios

**Scenario 1: Outbreak in a Rural Community**  
Farmers use the app to diagnose sick birds and receive treatment recommendations, enabling quick action and reducing losses.

**Scenario 2: Commercial Poultry Farm Management**  
Large farms use the system for daily health checks, enabling early detection and containment of diseases like New Castle Disease.

**Scenario 3: Research and Training for Veterinary Students**  
Veterinary students use the app for hands-on training, learning to diagnose and manage poultry diseases with modern technology.

---

## Folder Structure

```
Transfer-learning-based-poultry-disease-classification-system/
│
├── app.py                  # Main Flask application
├── chicken-disease.ipynb   # Model training and experimentation notebook
├── README.md               # Project documentation
│
├── model/
│   ├── chicken_model.keras # Final Keras model
│   └── efficientnetb3-Chicken-Disease.h5 # Original model
│
├── static/
│   ├── style.css           # Custom styles
│   └── assests/            # Images for UI
│       ├── image1.jpg
│       ├── image2.jpg
│       ├── image3.jpg
│       |── image4.jpg
|       |
│
├── templates/
│   └── index.html          # Main HTML template

```

---

## How to Run This Project

### 1. Clone the Repository

```
git clone <repo-url>
```

### 2. Set Up the Python Environment

- The project uses a virtual environment located in `tfenv/`.
- To activate it:
  - **Windows PowerShell:**
    ```
    .\tfenv\Scripts\Activate.ps1
    ```
  - **Windows CMD:**
    ```
    tfenv\Scripts\activate
    ```

### 3. Install Required Packages (if needed)

If you add new packages, install them using pip (while the environment is activated):

```
pip install -r requirements.txt
```

### 4. Run the Application

```
python app.py
```

- The app will start on `http://127.0.0.1:5000/` by default.
- Open this URL in your browser.

### 5. Use the Web App

- Click **GET STARTED**
- Upload a chicken image
- Click **Predict** to get the disease classification and confidence

---

## Credits

This project is an assignment from **SmartInternz**.

---

## Contact

For queries or contributions, please contact the project maintainer or SmartInternz.
