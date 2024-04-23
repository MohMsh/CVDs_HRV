import pandas as pd
import warnings
import os.path
from os import path

#Import from my code
#-------------------
import preprocess
import features
import svm
import logisticRegression as LR
import DeepNN
import AdaBoost
import XGBoost
import TabNet
import TabTransformers
import NODE

#Ignore all warnings
#-------------------
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

def DownloadPhysioNetData():
    print("Downloading Data")
    DownloadStatus = False
    print("Download service is not implemented yet")
    return DownloadStatus

def ExtractDataFromFolder():
    #Read from the data folder
    DataPath = "data/"
    
    #Read the number of records to iterate on
    RecordsFile = open(str(DataPath + "Records"),"r")
    Records = RecordsFile.readlines()
    RecordsFile.close()

    #Read the Cardiovascular Events from the Info File
    InfoFile = open(str(DataPath + "info.txt"),"r")
    Info = InfoFile.readlines()
    InfoFile.close()

    # A manual intervention is needed here
    # to correct the list of cardiovascular events
    # since the file containing the data is not indexed well
    # so a full automation is not achieved when getting the data
    CardioVascularEvents = []
    recordID,event = '',''
    index = 0

    for record in Info:
        if(index>0 and index<140):
            recordID = str("0"+str(record[0:4]))
            event = str(record[49:]).strip()
            #Change all Cardiovascular events into 1.0
            if(event == 'one' or event == 'ne' or
               event == 'a	none' or event == 'e' or event == 'none'):
                event = 0.0
            else:
                event = 1.0 
            CardioVascularEvents.append([recordID, event])
        index+=1
    df = pd.DataFrame(CardioVascularEvents, columns = ['recordID', 'CVEvent'])

    #Add empty columns to the DataFrame to match the HRV Features
    df['mean_nni']=''
    df['sdnn']=''
    df['sdsd']=''
    df['nni_50']=''
    df['pnni_50']=''
    df['nni_20']=''
    df['pnni_20']=''
    df['rmssd']=''
    df['median_nni']=''
    df['range_nni']=''
    df['cvsd']=''
    df['cvnni']=''
    df['mean_hr']=''
    df['max_hr']=''
    df['min_hr']=''
    df['std_hr']=''
    df['lf']=''
    df['hf']=''
    df['lf_hf_ratio']=''
    df['lfnu']=''
    df['hfnu']=''
    df['total_power']=''
    df['vlf']=''
    df['csi']=''
    df['cvi']=''
    df['Modified_csi']=''

    IndexesCount = 0
    
    #Preprocess the data and Extract HRV Features
    for record in Records:
        recordID = str(record).rstrip()

        #Get the raw data and preprocess it
        RRCorrected = preprocess.GetRecordData(str(DataPath + recordID))

        #Extract the HRV Features
        TimeDomainFeatures = features.get_time_domain_features(RRCorrected,True)
        FrequencyDomainFeatures = features.get_frequency_domain_features(RRCorrected)
        NonLinearDomainFeatures = features.get_csi_cvi_features(RRCorrected)

        RecordIndex = df.index[df['recordID'] == recordID].tolist()[0]

        #Add the HRV Features to the DataFrame                          
        for x,y in TimeDomainFeatures.items():
            df.at[RecordIndex,x] = y
        for x,y in FrequencyDomainFeatures.items():
            df.at[RecordIndex,x] = y
        for x,y in NonLinearDomainFeatures.items():
            df.at[RecordIndex,x] = y

        IndexesCount +=1
        print("Completed reading " + str(IndexesCount) + " out of "
              + str(len(Records)) + " record(s): "
              + str("{:.2f}".format(float(IndexesCount/len(Records))*100)) + "% completed.") 

    print("Features extraction phase has been completed")
    print("\n--------------------------------------------")
    SaveFeatures = input("Do you want to save the extracted features? (Y/N): ")
    if(SaveFeatures.upper() == "Y"):
        df = df.iloc[: , :]
        df.to_csv('ExtractedFeatures.csv', index=False)
    return df

def SelectSmartModel(MainDataFrame):
    print("\nChoose the SmartModel to be applied\n---------------------------------")
    print("A) Support Vector Machine (SVM) \nB) Logistic Regression (LR) \nC) Deep Neural Network (DNN) \nD) AdaBoost \nE) XGBoost \nF) DCNN - TabNet \nG) DCNN - TabTransformers \nH) DCNN - NODE")
    SmartModel = input("---------------------------------\n\nEnter Your choice: ")
    if(SmartModel.upper() == "A"):
        svm.ApplySVM(MainDataFrame)
    elif(SmartModel.upper() == "B"):
        LR.ApplyLogisticRegression(MainDataFrame)
    elif(SmartModel.upper() == "C"):
        DeepNN.ApplyDeepNN(MainDataFrame)
    elif(SmartModel.upper() == "D"):
        AdaBoost.ApplyAdaBoost(MainDataFrame)
    elif(SmartModel.upper() == "E"):
        XGBoost.ApplyXGBoost(MainDataFrame)
    elif(SmartModel.upper() == "F"):
        TabNet.ApplyTabNet(MainDataFrame)
    elif(SmartModel.upper() == "G"):
        MainDataFrame.columns = ["id","class","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        TabTransformers.ApplyTabTransformers(MainDataFrame)
    elif(SmartModel.upper() == "H"):
        MainDataFrame.columns = ["id","class","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
        NODE.ApplyNODE(MainDataFrame)
    else:
        print("Unrecognized Input. Try again")
        SelectSmartModel()
            
def MainMenu():
    print("\n\n***************************************************************")
    print("---------------------------------------------------------------")
    print("\nChoose From the below options (Use Q to quit):")
    print("---------------------------------")
    print("A) Read From Dataset (Apply all the preprocessig steps). \nB) Read From Prepared File (Skip all preprocessing steps)")
    print("---------------------------------")
    DataSource = input("\nEnter Your choice: ")
    MainDataFrame = pd.DataFrame()

    if(DataSource.upper() == "A"):
        if(path.exists("data/")!=True):
            print("Couldn't find the data folder")
            DownloadData = input("Download Data From PhysioNet? (Y/N): ")
            if(DownloadData.upper() == "Y"):
                DownloadStatus = DownloadPhysioNetData()

                if(DownloadStatus == True):
                    MainDataFrame = ExtractDataFromFolder()
                    SelectSmartModel(MainDataFrame)
                else:
                    print("Data was not downloaded successfully. Please retry!")
                    MainMenu()
            else:
                print("You choosed not to downlaod the data. You will be redirected to Main Menu.")
                MainMenu()
        else:
            print("Data is being read from the folder")
            MainDataFrame = ExtractDataFromFolder()
            SelectSmartModel(MainDataFrame)
    elif(DataSource.upper() == "B"):
        if(path.exists("ExtractedFeatures.csv")):
            MainDataFrame = pd.read_csv('ExtractedFeatures.csv')
            SelectSmartModel(MainDataFrame)
        else:
            print("Extracted features file was not found. You will be redirected to Main Menu.")
            MainMenu()
    elif(DataSource == "Q"):
        quit
    else:
        print("Unrecognized Input. You will be redirected to Main Menu.")
        MainMenu()

#The main code here
MainMenu()

input("\nProgram execution ended. Press Enter key to exit")
