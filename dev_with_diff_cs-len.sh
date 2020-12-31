cd /content/rerank_CSQA/
for i in {1..20}
do
    python train.py \
        --task_name $1 \
        --save_model_name $4 \
        --origin_model $2 \
        --cs_mode "QAconcept-Match" \
        --learning_rate 2e-5 \
        --cs_len 7 \
        --dev_cs_len $i \
        --output_dir $3 \
        --num_train_epochs 5 \
        --train_batch_size 4 \
        --eval_batch_size 2 \
        --gradient_accumulation_steps 5 \
        --fp16 \
        --check_loss_step 200 \
        --dev 
done