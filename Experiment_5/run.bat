@echo off

python Experiment_5_7.py ^
  --model_name_or_path ../pretrained_models/Qwen3-0.6B ^
  --train_file processed_data\train.json ^
  --validation_file processed_data\test.json ^
  --max_length 128 ^
  --per_device_train_batch_size 2 ^
  --learning_rate 2e-5 ^
  --num_train_epochs 3 ^
  --output_dir output