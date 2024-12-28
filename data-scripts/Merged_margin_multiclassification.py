
import pandas as pd
import argparse
def _sum (file_path: str , save: str):
    df  = pd.read_csv(file_path)
    df['final_target'] = df.loc[:,['marginal_text','merged']].sum(axis=1)

    df.to_csv(save)

    return "file saved"

def main(args):
    _sum(file_path=f'{args.data_folder}/train_data_stratified.csv', save= parser.save_folder + '/train_data_multi.csv')
    _sum(file_path=f'{args.data_folder}/test_set_stratified.csv', save= parser.save_folder + '/test_data_multi.csv')
    _sum(file_path=f'{args.data_folder}/val_set_stratified.csv', save= parser.save_folder + '/val_data_multi.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for multi-class text classification")
    parser.add_argument("--save_folder", type=str, default="data/multiclassification_data_set", help="Directory to save the trained model.")
    parser.add_argument('--data_folder' , type=str , default='data/data_stratified_sampling/' , help= "Directory to open the dataset")
    args = parser.parse_args()
    main(args)
