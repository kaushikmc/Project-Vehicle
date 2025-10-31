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
            # Returns a DataFrame suitable for model input (values are lists)
            vehicle_input_dict = self.get_vehicle_data_as_dict_list()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e

    def get_vehicle_data_as_dict_list(self):
        """
        This function returns a dictionary where values are lists (suitable for DataFrame creation).
        Used internally to create the input DataFrame.
        """
        logging.info("Entered get_vehicle_data_as_dict_list method of VehicleData class")
        try:
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
            
            # 1. Get raw input data for Gemini prompt (simple dict of single values)
            raw_input_data = {k: v[0] for k, v in dataframe.to_dict('list').items()}

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
            
            # --- CRITICAL FIX: Handle variable SHAP output structure ---
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Normal binary case: take the positive class (index 1) for the first sample (index 0)
                shap_values_list = shap_values[1][0] 
            elif isinstance(shap_values, list) and len(shap_values) == 1:
                # Biased case (returns only one class): take the only array available
                shap_values_list = shap_values[0][0]
            else:
                # Fallback (single array directly): use the array
                shap_values_list = shap_values[0]
            # ---------------------------------------------------------------------
            
            # 6. Create SHAP dictionary (map names to values)
            # .tolist() is used to convert the numpy array/list to a standard Python list, which is then handled
            # by the JSON serialization in main_utils.py
            raw_shap_dict = dict(zip(feature_names, shap_values_list.tolist() if hasattr(shap_values_list, 'tolist') else shap_values_list))
            
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
