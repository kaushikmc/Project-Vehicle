import os
import sys
import json
import numpy as np
import dill
import yaml
from pandas import DataFrame
from google import genai
import logging # Ensure logging is imported here

from src.exception import MyException
# from src.logger import logging # Assuming it's imported elsewhere or above

# Define environment key name
GEMINI_API_KEY_ENV_KEY = "GEMINI_API_KEY"

# --- Existing Utility Functions (read_yaml_file, save_object, etc.) ---

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise MyException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise MyException(e, sys) from e
# --- End of Existing Utility Functions ---


def get_gemini_explanation(input_features: dict, prediction: str, shap_values: dict) -> str:
    """
    Calls the Gemini API to generate a human-readable explanation
    for a model prediction based on feature values and SHAP scores.
    
    FIXED: Uses JSON serialization to handle complex NumPy types and passes clean strings to Gemini.
    """
    logging.info("Attempting to call Gemini API for prediction explanation.")
    try:
        # 1. Initialize Gemini Client
        gemini_api_key = os.getenv(GEMINI_API_KEY_ENV_KEY)
        if not gemini_api_key:
            logging.warning(f"Environment variable '{GEMINI_API_KEY_ENV_KEY}' not found. Cannot generate AI explanation.")
            return "Error: AI explanation service is not configured (API key missing)."

        client = genai.Client(api_key=gemini_api_key)
        
        # --- CRITICAL FIX: Explicit Type Conversion and JSON Prep ---
        # Convert input features to standard Python strings for the prompt
        cleaned_input_features = {k: str(v) for k, v in input_features.items()}
        
        # Prepare SHAP values: must be floats for JSON serialization
        json_serializable_shap_values = {}
        for k, v in shap_values.items():
            # If v is a list, take the first element (index 0) which is the contribution score.
            value = v[0] if isinstance(v, list) else v
            # Ensure the value is converted to a standard float before being used
            json_serializable_shap_values[k] = float(value) if value is not None else 0.0

        
        # 2. Construct the Detailed Prompt
        
        # Use JSON dump for the complex SHAP values, forcing all conversion issues here
        shap_json_string = json.dumps(json_serializable_shap_values, indent=2)
        
        feature_list = "\n".join([f"- {k}: {v}" for k, v in cleaned_input_features.items()])

        prompt = f"""
        Analyze the following machine learning prediction for a customer applying for vehicle insurance.
        The model made the prediction: **{prediction}**.

        Customer Input Features (Original Values):
        {feature_list}

        Feature Contributions (SHAP Values for Class 'Interested') as a JSON dictionary:
        {shap_json_string}
        
        Rules for Interpretation:
        1. A positive SHAP value pushed the prediction **TOWARDS 'Interested'**.
        2. A negative SHAP value pushed the prediction **TOWARDS 'Not Interested'**.
        3. The magnitude (absolute value) indicates influence strength.

        Generate a concise, single-paragraph explanation suitable for a business user.
        Identify the 2-3 most influential factors (features with the largest absolute SHAP values) and explain whether they increased or decreased the chance of a positive response. Conclude with a clear summary.
        """
        
        # 3. Call the API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        logging.info("Successfully received explanation from Gemini API.")
        # Replace Markdown formatting (like ** or *) that can interfere with HTML rendering
        return response.text.replace('**', '').replace('*', '') 

    except Exception as e:
        logging.error(f"Gemini API call failed: {e}", exc_info=True)
        # Return a user-friendly error message if the API call fails
        return f"Error generating human-readable explanation: {type(e).__name__} occurred. Details: {e}"
