cd /content/rerank_CSQA/
for i in {1...20}
do
    XLA_USE_BF16=1 python train.py \
        --task_name "rerank_csqa" \
        --save_model_name "Wsum_bert_base_2e-5_cslen7" \
        --origin_model "bert-base-cased" \
        --cs_mode "QAconcept-Match" \
        --learning_rate 2e-5 \
        --cs_len 7 \
        --dev_cs_len $1 \
        --output_dir "/content/drive/Shared drives/lab/rerank_CSQA/model_weight_sum" \
        --num_train_epochs 5 \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --gradient_accumulation_steps 5 \
        --fp16 \
        --check_loss_step 200 \
        --dev \
done