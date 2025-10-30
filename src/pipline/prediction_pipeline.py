import sys
import shap 
from pandas import DataFrame
from typing import Tuple, Optional, Dict

from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import get_gemini_explanation # Assumes this new function is in main_utils.py


class VehicleData:
    def __init__(self,
                 Gender,
                 Age,
                 Driving_License,
                 Region_Code,
                 Previously_Insured,
                 Annual_Premium,
                 Policy_Sales_Channel,
                 Vintage,
                 Vehicle_Age_lt_1_Year,
                 Vehicle_Age_gt_2_Years,
                 Vehicle_Damage_Yes
                 ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            # Note: These features are used to get the raw input dictionary for Gemini
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
            self.Vehicle_Damage_Yes = Vehicle_Damage_Yes

        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from VehicleData class input
        """
        try:
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Entered get_vehicle_data_as_dict method of VehicleData class")

        try:
            # Note: Values are wrapped in a list to create a DataFrame with one row
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_Year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_Years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_Yes]
            }

            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method of VehicleData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class VehicleDataClassifier:
    
    # Store the explainer and feature names across predictions
    _explainer = None
    _feature_names = None
    
    def __init__(self,prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.model_estimator = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
        except Exception as e:
            raise MyException(e, sys)

    def _get_or_init_explainer(self, model_object):
        """Initializes SHAP explainer and feature names once."""
        if VehicleDataClassifier._explainer is None:
            logging.info("Initializing SHAP explainer and retrieving feature names.")
            
            # The trained model object is the RandomForestClassifier
            VehicleDataClassifier._explainer = shap.TreeExplainer(model_object.trained_model_object)
            
            # Note: This is a placeholder list; actual feature count/order must be verified 
            # against your model's pipeline structure.
            VehicleDataClassifier._feature_names = [
                "Age (Standard Scaled)", 
                "Vintage (Standard Scaled)",
                "Annual_Premium (MinMax Scaled)",
                "Region_Code (MinMax Scaled)",
                "Policy_Sales_Channel (MinMax Scaled)",
                "Gender_Male",
                "Driving_License",
                "Previously_Insured",
                "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_gt_2_Years",
                "Vehicle_Damage_Yes"
            ]
            
        return VehicleDataClassifier._explainer, VehicleDataClassifier._feature_names

    def predict(self, dataframe: DataFrame) -> Tuple[int, Optional[Dict[str, float]], str]:
        """
        Predicts the class and generates a human-readable explanation using SHAP and Gemini.
        Returns: (prediction, raw_shap_dict, gemini_explanation)
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            
            # 1. Get raw input data for Gemini prompt
            raw_input_data = dataframe.iloc[0].to_dict()

            # 2. Load model and init XAI tools
            self.model_estimator.loaded_model = self.model_estimator.load_model()
            explainer, feature_names = self._get_or_init_explainer(self.model_estimator.loaded_model)

            # 3. Preprocess data
            preprocessed_data = self.model_estimator.loaded_model.preprocessing_object.transform(dataframe)
            
            # 4. Make prediction
            prediction = self.model_estimator.loaded_model.trained_model_object.predict(preprocessed_data)[0]
            prediction_label = "Interested" if prediction == 1 else "Not Interested"

            # 5. Generate SHAP values
            shap_values = explainer.shap_values(preprocessed_data)
            
            # --- CRITICAL FIX: Handle single-output vs. list-output SHAP values ---
            # If shap_values is a list (typical for multi-class/binary), take the values for class 1.
            # If it's a single numpy array, the model is too biased, and we use the first (and only) array output.
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Binary output: takes the values for the positive class (index 1)
                shap_values_array = shap_values[1][0] 
            elif isinstance(shap_values, list) and len(shap_values) == 1:
                 # Single output: This means the model is only predicting class 0, and SHAP only returned the values for that class. 
                 # We still use the only array available.
                shap_values_array = shap_values[0][0] 
            else:
                # If SHAP returns a single array directly (less common but possible for one input sample)
                shap_values_array = shap_values[0] 
            # ---------------------------------------------------------------------
            
            # 6. Create SHAP dictionary (map names to values)
            raw_shap_dict = dict(zip(feature_names, shap_values_array.tolist()))
            
            # 7. Generate Human-Readable Explanation from Gemini
            gemini_explanation = get_gemini_explanation(
                input_features=raw_input_data,
                prediction=prediction_label,
                shap_values=raw_shap_dict
            )
            
            logging.info("Exited predict method of VehicleDataClassifier class with XAI data.")
            return (prediction, raw_shap_dict, gemini_explanation)
        
        except Exception as e:
            # Log the exception before raising MyException
            logging.error(f"Prediction and XAI process failed: {e}", exc_info=True)
            # Return prediction 0 (default safe, conservative guess) and error message if prediction is mandatory
            return (0, None, f"An internal error occurred during prediction and explanation: {type(e).__name__}.")
