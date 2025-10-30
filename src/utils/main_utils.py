import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame
from google import genai # Already imported

from src.exception import MyException
from src.logger import logging


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
    file_path: str location of file to load
    return: Model/Obj
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
    file_path: str location of file to save
    array: np.array data to save
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
    file_path: str location of file to load
    return: np.array data loaded
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
    
    
GEMINI_API_KEY_ENV_KEY = "GEMINI_API_KEY"

def get_gemini_explanation(input_features: dict, prediction: str, shap_values: dict) -> str:
    """
    Calls the Gemini API to generate a human-readable explanation
    for a model prediction based on feature values and SHAP scores.
    
    FIXED: Converts all numerical dictionary values to standard Python types to prevent TypeError.
    """
    logging.info("Attempting to call Gemini API for prediction explanation.")
    try:
        # 1. Initialize Gemini Client
        gemini_api_key = os.getenv(GEMINI_API_KEY_ENV_KEY)
        if not gemini_api_key:
            logging.warning(f"Environment variable '{GEMINI_API_KEY_ENV_KEY}' not found. Cannot generate AI explanation.")
            return "Error: AI explanation service is not configured (API key missing)."

        client = genai.Client(api_key=gemini_api_key)
        
        # --- CRITICAL FIX: Explicit Type Conversion to avoid TypeError ---
        # Convert input features to standard Python strings for the prompt (str is most robust for heterogeneous data)
        cleaned_input_features = {k: str(v) for k, v in input_features.items()}
        
        # Convert SHAP values to standard Python floats for formatting (must succeed for a number)
        cleaned_shap_values = {k: float(v) for k, v in shap_values.items()}
        
        # ----------------------------------------------------------------------
        # DEBUG LOGGING BLOCK (Used for diagnosing the persistent TypeError)
        # ----------------------------------------------------------------------
        
        logging.debug("DEBUG: Data Types Before Gemini Call:")
        logging.debug(f"Input Feature Types: { {k: type(v) for k, v in cleaned_input_features.items()} }")
        logging.debug(f"SHAP Value Types: { {k: type(v) for k, v in cleaned_shap_values.items()} }")
        logging.debug(f"Input Feature Repr: {repr(cleaned_input_features)}")
        logging.debug(f"SHAP Value Repr: {repr(cleaned_shap_values)}")
        
        # ----------------------------------------------------------------------
        
        # 2. Construct the Detailed Prompt
        
        # Use cleaned dictionaries here
        feature_list = "\n".join([f"- {k}: {v}" for k, v in cleaned_input_features.items()])
        shap_list = "\n".join([f"- {k}: {v:.4f}" for k, v in cleaned_shap_values.items()])
        
        prompt = f"""
        Analyze the following machine learning prediction for a customer applying for vehicle insurance.
        The model made the prediction: **{prediction}**.

        Customer Input Features (Original Values):
        {feature_list}

        Feature Contributions (SHAP Values for Class 'Interested'):
        {shap_list}
        
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
        return f"Error generating human-readable explanation: {type(e).__name__} occurred."


# def drop_columns(df: DataFrame, cols: list)-> DataFrame:

#     """
#     drop the columns form a pandas DataFrame
#     df: pandas DataFrame
#     cols: list of columns to be dropped
#     """
#     logging.info("Entered drop_columns methon of utils")

#     try:
#         df = df.drop(columns=cols, axis=1)

#         logging.info("Exited the drop_columns method of utils")
        
#         return df
#     except Exception as e:
#         raise MyException(e, sys) from e
