cd /content/rerank_CSQA/
for i in {1..20}
do
    python train.py \
        --task_name "rerank_csqa" \
        --save_model_name $1 \
        --origin_model "bert-base-cased" \
        --cs_mode "QAconcept-Match" \
        --learning_rate 2e-5 \
        --cs_len 7 \
        --dev_cs_len $i \
        --output_dir "/content/drive/Shareddrives/lab/rerank_CSQA/model_weight_sum/" \
        --num_train_epochs 5 \
        --train_batch_size 4 \
        --eval_batch_size 2 \
        --gradient_accumulation_steps 5 \
        --fp16 \
        --check_loss_step 200 \
        --dev 
done