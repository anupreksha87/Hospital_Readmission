import logging_file
from Load_data import data_loader
from Preprocessing import Data_preprocessing
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd

class trainModel:

    def __init__(self):
        self.log_writer = logging_file.App_Logger()
        self.file_object = open("ModelTraining_log.txt", 'a+')

    def trainingModel(self,data=pd.read_csv("diabetic_data.csv")):
        # Logging the start of Training
        self.log_writer.log(self.file_object, '=========== Start of Training =============')
        try:
            # Getting the data from the source
            #data=pd.read_csv("diabetic_data.csv")
            """ doing the data preprocessing . 
            All the pre processing steps are based on the EDA done previously
            """
            """
            1. Initializing columns
            2. Changing null values to Other category in Condtion column
            3. Null removal
            4. Removing stopwords 
            5. Adding important Features 
            6. Vectorization 
            """
            # initializing preprocessor class
            preprocessor = Data_preprocessing.cleaner(self.file_object, self.log_writer)

            data = preprocessor.dropColumns(data)
            # removing null values in gender and race.
            data = preprocessor.replace_missing_values(data)
            data = preprocessor.replace_higher_null_values(data)
            # recategorize age
            data = preprocessor.recategorize_age(data)
            data = preprocessor.encode_target_variable(data)
            data = preprocessor.reduce_levels(data)
            data = preprocessor.encodeCategoricalValues(data)
            data = preprocessor.remove_skewness(data)
            data = preprocessor.feature_scaling(data)
            x = data[
                ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
                 'age']]
            y = data['readmitted']

            self.log_writer.log(self.file_object, 'X_cv_top_6,y have created sucessfully')

            X_cv_train, X_cv_test, Y_cv_train, Y_cv_test = model_selection.train_test_split(x, y,test_size=0.25,random_state=0)

            sm = SMOTE(random_state=20)
            print(x,y)

            train_input_new1, train_output_new2 = sm.fit_sample(X_cv_train, Y_cv_train)

            from sklearn.model_selection import train_test_split

            X_train, X_test, Y_train, Y_test = train_test_split(train_input_new1, train_output_new2,
                                                                                    test_size=0.20, random_state=100)
            rm2= RandomForestClassifier()
            rm2.fit(X_train,Y_train)
            file = open('pickle_files/rm2.pkl', 'wb')
            pickle.dump(rm2, file)

            self.log_writer.log(self.file_object, '=========== Training Succesfull =============')

        except Exception as e:
            self.log_writer.log(self.file_object,
                                   'Exception occured in creat_x_y method of the cleaner class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'creat_x_y is failed. Exited the creat_x_y method of the cleaner class')
            raise e


