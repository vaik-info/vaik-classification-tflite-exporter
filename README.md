# vaik-classification-tflite-exporter

Export from classification pb model to tflite model

## Usage

```shell
pip install -r requirements.txt
python export_tflite.py --input_model_dir_path ~/output_model/2022-11-07-15-55-32/step-1000_batch-8_epoch-9_loss_0.1008_sparse_categorical_accuracy_0.9715_val_loss_1.3999_val_sparse_categorical_accuracy_0.6440 \
                --train_input_dir_path ~/.vaik-mnist-classification-dataset/dump \
                --output_model_file_path ~/output_tflite_model/mnist_mobile_net_v2.tflite \
                --sample_max_num 25000
```

## Output

- ```~/output_tflite_model/mnist_mobile_net_v2.tflite```