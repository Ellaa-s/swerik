### Scripts related to Dataset to perform the following operations : 
  * Dataset Extraction : script to extract the data for certains years with random pages. 
   ```bash
          python3 script_dataextraction.py start-year --base_folder /path/to/dataset --output_folder /path/folder/to/be/saved --filer_per_folder number of files needed
   ```
   * Dataset creation : The below script are used to perform necesarry operations on extracted dataset for manual annotation.
   ```bash
         python3 filname.py
   ```
         The scripts available are : 
            - data_split.py : The dataset is large, hence this script can be used to split dataset. 
            - data_merge.py : The script is used to merge partitioned dataset that is annotated and save it in csv format as single file.
            - data_train_test_val.py : The script is used to split annotated dataset into train-val-test.
   
   * Class distribution : The below script are used to give the distribution of classes for margins and merged-margins.
      ```bash
            python3 dataset_classdistribution.py --data_folder <path to the data>
      ```
      ```bash
         python3 mergedmargins_sampling.py --data_folder <path to the data>
      ```
   *  Stratified sampling : The script used to create train-val-test with same class distribution for val and test data. This is used only if the class distribution count is not uniform for val and test data.
     ```bash
         python3 Dataset_test_stratifiysampling.py --data_folder 
```
   * Data exploration : This script is used to understand detailed information about the data.
    ```bash
         python3 Dataset_test_stratifiysampling.ipynb
```
   * Dataset for multiclassification : This script used to create dataset for multiclassifcation model.
       ```bash
         python3 Merged_margin_multiclassification.py --save_folder <path to save the folder> --data_folder <path to annotated dataset>
```  
### Scripts related fine tune the KB-Bert model with different approaches

*Baseline KB-Bert model
```bash
   python3 Bert_model_weighted_stratified.py --data_folder <path to the data> --save_folder <path to save output> --cuda <use if there is GPU> --save_predictions <path to save predictions> 
```
* Model fine-tuned with weights and stratified sampling for margin classification
```bash
   python3 Bert_model_weighted_stratified.py --data_folder <path to the data> --save_folder <path to save output> --cuda <use if there is GPU> --save_predictions <path to save predictions> --patience <the patience value for to trigger early stopping>
```
* Model fine-tuned with weights and stratified sampling for merged-margin classification
```bash
   python3 BertModel_mergedmargin_startified.py --data_folder <path to the data> --save_folder <path to save output> --cuda <use if there is GPU> --save_predictions <path to save predictions> --patience <the patience value for to trigger early stopping>
```

* Multiclassification model for classifying margins,merged-margins and other text
  ```bash
     python3 multiclassification.py --cude <to run with GPU> --save_folder <path to save the model>
  ```
* Active learning approach : The new dataset is extracted with 500 datapoints and is saved in data folder.
  ```bash
     python3 corrected_active_learning.py
  ```
  To test the model obatined from active learning approach, it can be run using below script on unseen dataset.
   ```bash
     python3 active_learning_testing.py
  ```
 * Fine-tuned with positional features :   

###Note : 
* Data : This folder consist of all files related to data. Below are description of subfolders in this.
           - data_partitioned : This contains dataset is obtained script data_split.py
           - data_stratied_sampling : This contains dataset train-val-set obtained after stratified sampling related to margin , merged-margins and positional features.
           - multiclassification_data_set : This contains dataset obtained from script Merged_margin_multiclassification.py used for multiclassifcation model.
           - old : This folder consist of data split 90-5-5 and dataset including years till the year 2022.
           - sampled_pages : It consist pages extracted including till the year 2022.

* Data_scripts : These folder consist of scripts used for dataset extraction and creation.
      

      
