import librosa
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, Wav2Vec2Processor

model_id = 'sakallana'
processor = Wav2Vec2Processor.from_pretrained(model_id)

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)


def prepare_dataset(batch):
    sr = 16000
    def load_mp3(file_path):
        au, _ = librosa.load(f'/kaggle/input/bengaliai-speech/train_mp3s/{file_path}.mp3', sr=sr)
        return au
    audio = list(map(load_mp3, batch['id']))
    # batched output is "un-batched"
    inputs = processor(audio, sampling_rate=sr)
    inputs["input_values"] = inputs.input_values
    inputs["labels"] = processor(text=batch["sentence"]).input_ids
    return inputs

class DataCollatorWav2txtWithPadding:
    def __init__(self, processor, padding = True, max_length = None, max_length_labels = None,
                 pad_to_multiple_of = None, pad_to_multiple_of_labels = None, return_tensors='pt'):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels
        self.return_tensors = return_tensors
    
    def __call__(self, features, return_tensors=None):
        if return_tensors is not None:
            self.return_tensors = return_tensors
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features,
                                    padding=self.padding,
                                    max_length=self.max_length,
                                    pad_to_multiple_of=self.pad_to_multiple_of,
                                    return_tensors=self.return_tensors)
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features,
                                              padding=self.padding,
                                              max_length=self.max_length_labels,
                                              pad_to_multiple_of=self.pad_to_multiple_of_labels,
                                              return_tensors=self.return_tensors)
        
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
        

from datasets import load_dataset

data = load_dataset('data', streaming=False)


data['train'] = data['train'].filter(lambda example, indice: indice % 12 == 0, with_indices=True)
data['validation'] = data['validation'].filter(lambda example, indice: indice % 10 == 0, with_indices=True)

maped_data = data.shuffle(seed=40).map(prepare_dataset, num_proc=25, batched=True, batch_size=50, remove_columns=['id', 'sentence'], keep_in_memory=True)
        


from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load

colleter = DataCollatorWav2txtWithPadding(processor, padding='longest')
wer = load("wer")

def train():
    def wer_metric(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(2)
        predictions = processor.tokenizer.batch_decode(predictions)
        labels = processor.tokenizer.batch_decode(labels)
        return wer.compute(predictions=predictions, references=labels,)
    
    args = Seq2SeqTrainingArguments(
                                    output_dir=model_id,
                                    do_train=True,
                                    do_eval=True,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    gradient_accumulation_steps=20,
                                    learning_rate=3e-3,
                                    weight_decay=0.1,
                                    max_grad_norm=0.3,
                                    num_train_epochs=1,
                                    warmup_ratio=0.2,
                                    dataloader_drop_last=True,
                                    sortish_sampler=True,
#                                     group_by_length=True,
                                    torch_compile=True,
                                    save_steps=500,
                                    eval_delay=500,
                                    label_smoothing_factor=0.1,
                                    evaluation_strategy='steps', tpu_num_cores=8,
                                    predict_with_generate=True,
                                    run_name='test_tpu')
    
    
    doing = Seq2SeqTrainer(model, args, data_collator=colleter, 
                           train_dataset=maped_data['train'],
                           eval_dataset=maped_data['validation'],
                           compute_metrics=wer_metric
                          )
    
    doing.train()
    
if __name__ == '__main__':
    train()
