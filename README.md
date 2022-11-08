# vaik-classification-tflite-exporter

Export from classification pb model to tflite model

## Usage

```shell
pip install -r requirements.txt
python export_tflite.py --input_model_dir_path ~/output_model \
                --train_input_dir_path ~/.vaik-mnist-classification-dataset/dump \
                --output_model_file_path ~/output_tflite_model/mnist_mobile_net_v2.tflite \
                --sample_max_num 25000
```

## Output

- ```~/output_tflite_model/mnist_mobile_net_v2.tflite```