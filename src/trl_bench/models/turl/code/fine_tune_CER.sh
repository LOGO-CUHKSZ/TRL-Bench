OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam
SEED=0
CUDA_VISIBLE_DEVICES="0" python run_table_CER_finetuning.py \
    --output_dir=output/CER/v2/$OUTPUT_DIR"_seed_"$SEED"_10000" \
    --model_name_or_path=output/hybrid/v2/$OUTPUT_DIR \
    --model_type=CER \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --evaluate_during_training \
    --per_gpu_train_batch_size=20 \
    --per_gpu_eval_batch_size=20 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --num_train_epochs=10 \
    --save_total_limit=10 \
    --seed_num=$SEED \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --max_entity_candidate=10000 \
    --config_name=configs/table-base-config_v2.json \
    --save_steps=5000 \
    --logging_steps=1000 \
    --use_cand > output/CER/v2/$OUTPUT_DIR"_seed_"$SEED"_10000"/train.log 2>&1 &