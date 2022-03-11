from utils import process_csv, folder_with_time_stamps
import training_params
from dataset import SentimentDataset
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import wandb

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


log_folder, checkpoint_folder = folder_with_time_stamps(training_params.LOG_DIR,
                                                        training_params.CHECKPOINT_DIR)
os.makedirs(log_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)

writer = SummaryWriter(log_folder)
config = {
  "learning_rate": training_params.LEARNING_RATE,
  "batch_size": training_params.BATCH_SIZE,
  "num_epochs" : training_params.EPOCHS,
  "max_len" : training_params.MAX_LEN,
  "load_checkpoint" : training_params.LOAD_CHECKPOINT,
  "max_grad_norm" : training_params.MAX_GRAD_NORM
}


run = wandb.init(project=training_params.WANDB_PROJECT_NAME, job_type='train_model' ,config=config)
cfg = wandb.config


train_texts, train_targets = process_csv(training_params.TRAIN_DATA)
valid_texts, valid_targets = process_csv(training_params.VALID_DATA)

# # subset

# indices_train = random.sample(range(len(train_texts)), 1000)
# indices_valid = random.sample(range(len(valid_texts)), 500)

# train_texts = [t for i, t in enumerate(train_texts) if i in indices_train]
# train_targets = [t for i, t in enumerate(train_targets) if i in indices_train]
# valid_texts = [t for i, t in enumerate(valid_texts) if i in indices_valid]
# valid_targets = [t for i, t in enumerate(valid_targets) if i in indices_valid]

train_dataset = SentimentDataset(texts=train_texts, targets=train_targets)
valid_dataset = SentimentDataset(texts=valid_texts, targets=valid_targets)

# d = train_dataset.__getitem__(0)
# print(d.keys())
# print(d['input_ids'].shape)
# print(d['attention_mask'].shape)
# print(d['targets'].shape)

train_data_loader = DataLoader(train_dataset, batch_size=training_params.BATCH_SIZE, num_workers=4)
valid_data_loader = DataLoader(valid_dataset, batch_size=training_params.BATCH_SIZE, num_workers=4)

model = BertForSequenceClassification.from_pretrained(
                                                        training_params.PRE_TRAINED_MODEL_NAME,
                                                        num_labels=len(training_params.LABEL_DICT),
                                                        output_attentions=False,
                                                        output_hidden_states=False
                                                    )

if training_params.FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = training_params.LEARNING_RATE,
    eps = 1e-8
)

total_steps = len(train_data_loader) * training_params.EPOCHS
epochs = training_params.EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps = total_steps
)

starting_epoch = 0
# if training_params.LOAD_CHECKPOINT:
#     checkpoint = torch.load(training_params.CHECKPOINT_PATH)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])

#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if torch.is_tensor(v):
#                 state[k] = v.to(training_params.DEVICE)

#     starting_epoch = checkpoint['epoch'] + 1


model.to(training_params.DEVICE)

loss_values, validation_loss_values = [], []

wandb.watch(model)
# training loop
train_step_count = 0
for epoch in range(starting_epoch, training_params.EPOCHS):

    model.train()
    total_loss = 0

    # Training loop
    tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)), unit='batch')
    tk0.set_description(f'Epoch {epoch + 1}')

    for step, batch in enumerate(tk0):
        #  add batch to gpu
        for k, v in batch.items():
            batch[k] = v.to(training_params.DEVICE)

        b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['targets']

        model.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        
        loss = outputs['loss']
        loss.backward()
        total_loss += loss.item()

        
        # loss for step
        writer.add_scalar("Training Loss - Step", loss.sum(), train_step_count)
        run.log({'Training Loss - Step': loss.sum()})

        train_step_count += 1
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=training_params.MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_data_loader)
    print("Average train loss: {}".format(avg_train_loss))

    writer.add_scalar("Training Loss", avg_train_loss, epoch)
    run.log({'Training loss': avg_train_loss, 'epoch': epoch})

    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_folder + '/checkpoint_last.pt')
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    best_val_loss = np.inf

    for batch in tqdm(valid_data_loader, total=int(len(valid_data_loader)), unit='batch', leave=True):
        for k, v in batch.items():
            batch[k] = v.to(training_params.DEVICE)
        b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['targets']

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        logits = outputs['logits'].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs['loss'].item()
        
        preds = np.argmax(logits, axis=1)
        predictions.extend(preds)
        true_labels.extend(label_ids)
        
#         print(predictions)
#         print(true_labels)

    eval_loss = eval_loss / len(valid_data_loader)

    if eval_loss < best_val_loss:
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, checkpoint_folder + '/checkpoint_best.pt')
        best_val_loss = eval_loss

    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))

    writer.add_scalar("Validation Loss", eval_loss, epoch)
    run.log({'Validation loss': eval_loss})

    val_accuracy = accuracy_score(true_labels, predictions)
    val_f1_score = f1_score(true_labels, predictions, average='macro')
    true_labels_names = [training_params.LABEL_DICT[str(tl)] for tl in true_labels]
    predictions_names = [training_params.LABEL_DICT[str(pr)] for pr in predictions]
    report = classification_report(true_labels_names, 
                                        predictions_names, output_dict=True,
                                        labels=np.unique(predictions_names))

    df_report = pd.DataFrame(report).transpose() 
    df_report['categories'] = list(df_report.index) 
    df_report = df_report[ ['categories'] + [ col for col in df_report.columns if col != 'categories' ] ] 
    classification_table = wandb.Table(dataframe=df_report) 
    run.log({f'Confusion Matrix Epoch {epoch}': classification_table})

    print("Validation Accuracy: {}".format(val_accuracy))
    print("Validation F1-Score: {}".format(val_f1_score))
    print("Classification Report: {}".format(report))
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
    writer.add_scalar('Validation F1 score', val_f1_score, epoch)
    run.log({'Validation Accuracy': val_accuracy})
    run.log({'Validation F1 Score': val_f1_score})

    
    
run.finish()