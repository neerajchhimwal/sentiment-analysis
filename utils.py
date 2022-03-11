import pandas as pd 
import training_params
import datetime

def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    texts = df['clean_title'].values
    targets = df['preds'].values
    # print(max(df.title_word_count.values))
    # print(df.preds.value_counts(normalize=True))
    return texts, targets

def folder_with_time_stamps(log_folder, checkpoint_folder):
    folder_hook = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_saving = log_folder + '/' + folder_hook
    checkpoint_saving = checkpoint_folder + '/' + folder_hook
    # train_encoder_file_path = '/'.join(training_params.TRAIN_DATA.split('/')[:-1]) + '/label_encoder_' \
    #                           + folder_hook + '.json'
    return log_saving, checkpoint_saving #, train_encoder_file_path, 

if __name__ == "__main__":
    csv = '~/Downloads/sentiment_data/df_train_with_preds.csv'
    text_list, preds_list = process_csv(csv)