### Scripts related to the Dataset  
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
            - data_pos_features.py: The script gets the positional features and saves them in a new file, uses functions from the prewritten script get_positional_features.py.
   
   * Class distribution : The below scripts are used to give the distribution of classes for margins and merged-margins.
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
   * Data exploration : These scripts are used to understand detailed information about the data.
     ```bash
         - Dataset_test_stratifiysampling.ipynb
         - data_exploration.ipynb
         - plot_pos_features.ipynb
      ```
   * Dataset for multiclassification : This script used to create dataset for multiclassifcation model.
     ```bash
         python3 Merged_margin_multiclassification.py --save_folder <path to save the folder> --data_folder <path to annotated dataset>
     ```  

### Scripts to fine tune the KB-Bert model with different approaches

* Baseline KB-Bert model
```bash
   python3 Bert_train.py --data_folder <path to the data> --save_folder <path to save output> --cuda <use if there is GPU> --save_predictions <path to save predictions> 
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
 * Fine-tuned with positional features: The BERT model with positional features is based on pretrained models.
1. Download the model from the release KBBertmodel_stratifiedsampling_withweights and save it in the folder "./output/output/margin_prediction_model/". This model is trained by the script "Bert_model_weighted_stratified.py".
2. Download the model from the release Positional_FFNN and save it in a folder in the repository. This model is trained by the script "network_pos_features.py".
3. Run the following command to train the model combining the two pretrained models.
```bash
  python3 .\Bert_pos_features.py --data_folder ".\data\" --save_folder <path to save the model> --model_folder <path where the positional_ffnn model is saved> --save_predictions <if the resulting predictions should be saved>
```
### Note : 
- **Data** : This folder consists of all files related to the data. Below is a description of the subfolders contained within:
    - **data_partitioned**: Contains the dataset obtained from the script `data_split.py`.
    - **data_stratified_sampling**: Contains the train-validation dataset obtained after stratified sampling based on margin, merged- margins, and positional features.
    - **multiclassification_dataset**: Contains the dataset generated by the script `Merged_margin_multiclassification.py`, used for the multiclassification model.
    - **old**: This folder consists of the 90-5-5 data split and datasets that include data up to the year 2022.
    - **sampled_pages**: Contains the extracted pages, including data up to the year 2022.


- **Data_scripts** : These folder consist of scripts used for dataset extraction and creation.
- **Models** : The models can be downloaded from the release page.
      

      
