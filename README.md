# Wine Classifier API

## Overview
The **Wine Classifier API** is a machine learning-powered RESTful service built with **FastAPI**. It predicts the wine class (target) based on chemical features of the wine. The model uses a Logistic Regression classifier trained on the popular **Wine dataset** from scikit-learn.

## Features
- Input validation using **Pydantic** for robust and reliable API behavior.
- Logistic Regression model to classify wine into three categories (0, 1, 2).
- Scalable API with FastAPI, providing built-in Swagger UI for testing.
- Model persistence using **Pickle** for reusability.
- Supports JSON-based requests for seamless integration.

---

## Endpoints

### **POST** `/predict`

#### Request
Send a JSON payload containing the following features:

| Feature                         | Type  | Description                        |
|---------------------------------|-------|------------------------------------|
| `alcohol`                      | float | Alcohol content in the wine.       |
| `malic_acid`                   | float | Malic acid content.                |
| `ash`                          | float | Ash content.                       |
| `alcalinity_of_ash`            | float | Alkalinity of ash.                 |
| `magnesium`                    | float | Magnesium level.                   |
| `total_phenols`                | float | Total phenols content.             |
| `flavanoids`                   | float | Flavanoids content.                |
| `nonflavanoid_phenols`         | float | Non-flavanoid phenols content.     |
| `proanthocyanins`              | float | Proanthocyanins level.             |
| `color_intensity`              | float | Color intensity of the wine.       |
| `hue`                          | float | Hue measurement.                   |
| `od280_od315_of_diluted_wines` | float | OD280/OD315 ratio of diluted wines.|
| `proline`                      | float | Proline content.                   |

#### Example Request
```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100.0,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050.0
}
```

#### Response
The API responds with a JSON object containing the predicted wine class (0, 1, or 2).

```json
{
  "prediction": 1
}
```

---

## Project Structure

```plaintext
.
├── app.py                     # Main FastAPI application
├── model/
│   ├── classifier.pkl         # Pre-trained logistic regression model
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

---

## Installation

### Prerequisites
- Python 3.8 or above.
- Virtual environment (optional but recommended).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/wine-classifier-api.git
   cd wine-classifier-api
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the API:
   ```bash
   uvicorn app:app --reload
   ```

5. Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Model Training
The model was trained using the **Wine dataset** from scikit-learn. For details on training and preprocessing, refer to the `train_model` function in the script.

### Steps to Retrain
1. Load the Wine dataset using scikit-learn.
2. Preprocess the data (scaling and splitting).
3. Train a Logistic Regression model.
4. Save the model using `pickle`.

---

## Testing the API
You can test the API using tools like:
- **Swagger UI**: Automatically available at `/docs` when the API is running.
- **Postman**: Use the provided example request.
- **Curl**: Command-line tool for API testing.

### Example with Curl
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"alcohol":13.2,"malic_acid":1.78,"ash":2.14,"alcalinity_of_ash":11.2,"magnesium":100.0,"total_phenols":2.65,"flavanoids":2.76,"nonflavanoid_phenols":0.26,"proanthocyanins":1.28,"color_intensity":4.38,"hue":1.05,"od280_od315_of_diluted_wines":3.4,"proline":1050.0}'
```

---

## Enhancements
- Add more advanced models for comparison.
- Implement better error handling for edge cases.
- Add support for model versioning.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
