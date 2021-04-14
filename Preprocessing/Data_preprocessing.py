import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import preprocessing


class cleaner:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object


    def replaceInvalidValuesWithNull(self, data):

        """
                               Method Name: is_null_present
                               Description: This method replaces invalid values i.e. '?' with null, as discussed in EDA.



                                       """
        self.logger_object.log(self.file_object,
                               'Entered the replaceInvalidValuesWithNull method of the Model_Finder class')
        try:



            for column in data.columns:

                count= data[column][data[column] == '?'].count()
                if count != 0:
                    data[column] = data[column].replace('?', np.nan)
            self.logger_object.log(self.file_object,
                               'replace "?" with nan is  success. Exited the is_null_present method of the Preprocessor class')

            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in replaceInvalidValuesWithNull method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'replaceInvalidValuesWithNullvalues failed. Exited the replaceInvalidValuesWithNull method of the Preprocessor class')
            raise Exception()


    def is_null_present(self, data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                        returns the list of columns for which null values are present.
                                On Failure: Raise Exception



                        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            self.null_counts = data.isna().sum()  # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if (self.null_present):  # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv(
                    'preprocessing_data/null_values.csv')  # storing the null column information to file
            self.logger_object.log(self.file_object,
                                   'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the cleaner class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the cleaner class')
            raise Exception()



    def dropColumns(self, data):

            """
                            Method Name: dropColumns
                            Description: This method drops the unwanted columns as discussed in EDA section.



                                    """
            self.logger_object.log(self.file_object, 'Entered the dropColumns method of the cleaner class.')
            try:

                data.drop(['citoglipton', 'examide', 'encounter_id', 'patient_nbr', 'weight', 'diag_2', 'diag_3',
                              'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride','repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                              'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                              'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                              'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                              'metformin-rosiglitazone', 'metformin-pioglitazone', "max_glu_serum", "A1Cresult","diag_1"],axis=1, inplace=True)

                self.logger_object.log(self.file_object,'drop column is a success. Exited the drop_column method of the cleaner class')

                return data
            except Exception as e:
                self.logger_object.log(self.file_object,
                                       'Exception occured in drop_columns method of the cleaner class. Exception message:  ' + str(
                                           e))
                self.logger_object.log(self.file_object,
                                       'drop columns is failed. Exited the drop_column method of the cleaner class')
                raise e

    def replace_missing_values(self, data):
        """
                                        Method Name: replace_missing_values
                                        Description: This method replaces all the missing values in those columns in which missing values are completely at random , with mode.
                                        Output: A Dataframe which has all the missing values imputed.

                     """
        data["race"] = data["race"].replace("?", "Caucasian")
        data["gender"]=data["gender"].replace("Unknown/Invalid","Female")

        return data


    def replace_higher_null_values(self,data) :
        """ this method replace the null values as a diff category .and one hot encoding only for top 6 frequency"""
        "FOR PAYER CODE."
        self.logger_object.log(self.file_object, 'Entered the replace_higher_null_values  method of the cleaner class.')
        try:

            self.lst_10 = data.payer_code.value_counts().sort_values(ascending=False).head(6).index[0:]
            self.lst_10 = list(self.lst_10)
            for categories in self.lst_10:
               data[categories] = np.where(data["payer_code"] == categories, 1, 0)
            self.lst_10.append("payer_code")
            self.logger_object.log(self.file_object,
                                   'replace null_values of payer_code is a succes of the cleaner class')

            "FOR MEDICAL_SPECIALTY"
            self.lst_10 = data.medical_specialty.value_counts().sort_values(ascending=False).head(10).index[0:]
            self.lst_10 = list(self.lst_10)
            for categories in self.lst_10:
                data[categories] = np.where(data["medical_specialty"] == categories, 1, 0)
            self.lst_10.append("payer_code")
            " droping medical_specialty and payer_code"
            data.drop(["medical_specialty", "payer_code"], axis=1, inplace=True)
            self.logger_object.log(self.file_object,
                               'replace_higher_null_values is a success. Exited the replace_higher_null_values method of the cleaner class')

            return data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in replace_higher_null_values method of the cleaner class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'replacing_higher_null_values is failed. Exited the replace_higher_null_values method of the cleaner class')
            raise Exception()

    def recategorize_age(self,data):
        "   This method will recategorize the age feature.   "
        self.logger_object.log(self.file_object, 'Entered the recategorize_age  method of the cleaner class.')
        try:


            for i in range(0, 10):

                data['age'] = data['age'].replace('[' + str(10 * i) + '-' + str(10 * (i + 1)) + ')', i + 1)
            age_dict = {1: 5, 2: 15, 3: 25, 4: 35, 5: 45, 6: 55, 7: 65, 8: 75, 9: 85, 10: 95}
            data['age'] = data.age.map(age_dict)
            self.logger_object.log(self.file_object,
                                  'recategorize is a success . Exited the recategorize_age method of the cleaner class')

            return data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in recategorize_age method of the cleaner class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'recategorize_age is failed. Exited the recategorize_age method of the cleaner class')
            raise Exception()


    def encode_target_variable(self,data):
        " This method will replace the targrt variable with 0 or 1."
        self.logger_object.log(self.file_object, 'Entered the encode_target_variable  method of the cleaner class.')
        try:

            #We can see that there are three types of values:- 'NO','<30','>30'. So, lets make it a categorical variable
            " '0' means NO readmission,and readmission after 30days and '1' means readmission in 30 days after patient discharged."
            data['readmitted'] = data['readmitted'].replace('NO', 0)
            data['readmitted'] = data['readmitted'].replace('<30', 1)
            data['readmitted'] = data['readmitted'].replace('>30', 0)
            self.logger_object.log(self.file_object,
                               'encode_target_variable is a success. Exited the encode_target_variable method of the cleaner class')

            return data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in encode_target_variable method of the cleaner class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'encoding of target variable is  failed. Exited the encode_target_variable method of the cleaner class')
            raise Exception()



    def reduce_levels(self,data):
        self.logger_object.log(self.file_object, 'Entered the reduce_levels  method of the cleaner class.')
        try:

            " This method will reduce the level of 'discharge_disposition_id', 'admission_source_id','admission_type_id' ."
            # original 'discharge_disposition_id' contains 28 levels, so reducing levels into 2 categories
            data['discharge_disposition_id'] = [0 if val == 1 else 1
                                          for val in data['discharge_disposition_id']]
            # original 'admission_source_id' contains 25 levels so reducing it into 3 categories
            data['admission_source_id'] = [0 if val == 7 else 1 if val == 1 else 2
                                     for val in data['admission_source_id']]
            # original 'admission_type_id' contains 8 levels, so reducing it into 2 categories
            data['admission_type_id'] = [0 if val == 1 else 1
                                   for val in data['admission_type_id']]
            # Denote 'diag_1' as '1' if it relates to diabetes and '0' if it's not.

            self.logger_object.log(self.file_object,
                                  'reduce_levels is a success. Exited the reduce_levels method of the cleaner class')

            return data

        except Exception as e:
            self.logger_object.log(self.file_object,
                           'Exception occured in reduce_levels method of the cleaner class. Exception message:  ' + str(
                               e))
            self.logger_object.log(self.file_object,
                           'reduce_levels is failed. Exited the reduce_levels method of the cleaner class')
            raise Exception()

    def encodeCategoricalValues(self,data):
        self.logger_object.log(self.file_object, 'Entered the encodecategoricalvalues method of the cleaner class.')
        try:
            self.keys = ["insulin", "metformin"]
            for col in self.keys:



                data[col] = data[col].replace('No', 0)
                data[col] = data[col].replace('Steady', 1)
                data[col] = data[col].replace('Up', 1)
                data[col] = data[col].replace('Down', 1)

            data['change'] = data['change'].replace('Ch', 1)
            data['change'] = data['change'].replace('No', 0)
            data['gender'] = data['gender'].replace('Male', 1)
            data['gender'] = data['gender'].replace('Female', 0)
            data['diabetesMed'] = data['diabetesMed'].replace('Yes', 1)
            data['diabetesMed'] = data['diabetesMed'].replace('No', 0)

            self.just_dummies = pd.get_dummies(data['race'], drop_first=True)
            data = pd.concat([data, self.just_dummies], axis=1)
            data.drop(['race'], axis=1, inplace=True)


            self.logger_object.log(self.file_object,
                       'encode_target_variable is a success. Exited the encode_target_variable method of the cleaner class')

            return data

        except Exception as e:
            self.logger_object.log(self.file_object,
                       'Exception occured in reduce_levels method of the cleaner class. Exception message:  ' + str(
                           e))
            self.logger_object.log(self.file_object,
                       'reduce_levels is failed. Exited the reduce_levels method of the cleaner class')
            raise Exception()


    def remove_skewness(self,data):
         # Applying square root transformation on right skewed count data to reduce the effects of extreme values.
        self.logger_object.log(self.file_object, 'Entered the reduce_skewness method of the cleaner class.')
        try:

            data['number_outpatient'] = data['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
            data['number_emergency'] = data['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
            data['number_inpatient'] = data['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))
            self.logger_object.log(self.file_object,
                                'remove_skewness is a success. Exited the remove_skewness method of the cleaner class')

            return data

        except Exception as e:
            self.logger_object.log(self.file_object,
                       'Exception occured in encode_categorical_value of the cleaner class. Exception message:  ' + str(
                           e))
            self.logger_object.log(self.file_object,
                       'encoding categorical value is failed. Exited the encode categorical_value method of the cleaner class')
            raise Exception()


    def feature_scaling(self,data):
        self.logger_object.log(self.file_object, 'Entered the feature_scaling method of the cleaner class.')
        try:


            feature_scale_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                          'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient']

            scaler = preprocessing.StandardScaler().fit(data[feature_scale_cols])
            data_scaler = scaler.transform(data[feature_scale_cols])

            data_scaler_df = pd.DataFrame(data=data_scaler, columns=feature_scale_cols, index=data.index)
            data.drop(feature_scale_cols, axis=1, inplace=True)
            data = pd.concat([data, data_scaler_df], axis=1)
            self.logger_object.log(self.file_object, 'Entered the feature_scaling method of the cleaner class.')

            return data

        except Exception as e:
            self.logger_object.log(self.file_object,
                       'Exception occured in feature_scaling method of the cleaner class. Exception message:  ' + str(
                           e))
            self.logger_object.log(self.file_object,
                       'feature_scaling is failed. Exited the feature_scaling method of the cleaner class')
            raise e


    def create_x_y(self,data):

         # Creating X (features) and y (response)
        self.logger_object.log(self.file_object, 'Entered the creat_x_y method of the cleaner class.')
        try:
            self.X_cv_top6 = data[
              ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
               'age']]
            self.y = data['readmitted']

            self.logger_object.log(self.file_object, 'success the feature_scaling method of the cleaner class.')
            return self.X_cv_top6,self.y
        except Exception as e:
            self.logger_object.log(self.file_object,
                       'Exception occured in creat_x_y method of the cleaner class. Exception message:  ' + str(
                           e))
            self.logger_object.log(self.file_object,
                       'creat_x_y is failed. Exited the creat_x_y method of the cleaner class')
            raise Exception()


    def feature_importance(self,x,y):
       from sklearn.ensemble import ExtraTreesRegressor

       self.model = ExtraTreesRegressor()
       self.model.fit(self.x,self.y)

    def top_feature(self,data):
        #Finding top 6 features.
        feat_importances = pd.Series(self.model.feature_importances_, index=self.x.columns)
        top_6_feature=feat_importances.nlargest(6)
        print(top_6_feature)
        self.X_cv_top6 = data[['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
           'age']]
        return self.X_cv_top6








