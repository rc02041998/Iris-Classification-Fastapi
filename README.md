# Iris Classification-Fastapi
 

## Overview
This is a simple FastAPI-based web service for making predictions using a pre-trained machine learning model. The model is trained on the Iris dataset and can classify flowers into one of three categories: `setosa`, `versicolor`, or `virginica`.

## Requirements
Ensure you have the following dependencies installed before running the application:

```bash
pip install fastapi pydantic numpy scikit-learn uvicorn
```

## Project Structure
```
.
├── model.pkl  # Pre-trained machine learning model
├── main.py    # FastAPI application
├── README.md  # Documentation
```

## Running the Application
To start the FastAPI server, run the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Home
- **Endpoint:** `/`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "message": "ML Model API is running"
  }
  ```

### Prediction
- **Endpoint:** `/predict`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "features": [5.1, 3.5, 1.4, 0.2]
  }
  ```
- **Response:**
  ```json
  {
    "prediction": "setosa"
  }
  ```

## Error Handling
If the input format is incorrect or an exception occurs, the API will return an error response:
```json
{
  "error": "Some error occured"
}
```

## Notes
- The model is loaded from `model.pkl`.
- Ensure the input features match the expected format (a list of four floating-point numbers).

## Author
- Developed by Rohit Kumar

