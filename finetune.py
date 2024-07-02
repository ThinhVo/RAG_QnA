import os
import openai  # !pip install openai==0.27.9

import pandas as pd
import io
from matplotlib import pyplot as plt

from dotenv import load_dotenv
load_dotenv()

import time

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Upload training and validation JSONL file to openai

train_res = openai.files.create(
    file=open(os.path.join('data', "QA_pairs_train.jsonl"), 'rb'),
    purpose='fine-tune'
)
train_file_id = train_res.id

val_res = openai.files.create(
    file=open(os.path.join('data', "QA_pairs_val.jsonl"), 'rb'),
    purpose='fine-tune'
)
val_file_id = val_res.id

# Create finetune job
finetune_job = openai.fine_tuning.jobs.create(
    training_file=train_file_id,
    validation_file=val_file_id,
    model="gpt-3.5-turbo-1106",
    hyperparameters={'n_epochs': 4, 'batch_size': 3}
)
ft_id = finetune_job.id

# Check finetune job status
while True:
    ftr = openai.fine_tuning.jobs.retrieve(fine_tuning_job_id=ft_id)
    print(ftr.status)
    if ftr.finished_at:
        fine_tuned_model = ftr.fine_tuned_model
        break
    time.sleep(60)

result_files = ftr.result_files

metrics = openai.files.content(file_id=result_files[0])

df_metrics = pd.read_csv(io.StringIO(metrics.content.decode('UTF-8')))
df_metrics.to_csv(os.path.join(f'finetune_result_{result_files}.csv'), index=False)

with open('finetune_job_id.txt', 'w') as f:
    f.write(ft_id)