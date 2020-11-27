# charCNN-BERT-CRF
The model combines character CNN, BERT and CRF and aims at clinical de-identification based on Named Entity Recognition (NER).

First you must download BERT: BERT-Base, Multilingual Cased (New, recommended).
And then reset your root path and bert path in main.py.
Unzip your data in root path, and set the data dir in main(_) of main.py. If you wanna i2b2 2014 deId data, mail me anyway.

python main.py \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --max_seq_length=128 \
  --num_train_epochs=5.0 \
  --output_dir=/root/output/
  
  The overall architecture is as follows.
  ![Image text](https://github.com/Zhengxuru/charCNN-BERT-CRF/blob/master/charcnn_bert_crf.png)
  
  The architecture of CharCNN is as follows. 
  ![Image text](https://github.com/Zhengxuru/charCNN-BERT-CRF/blob/master/charCNN.png)
  
 You can shut down the character CNN or CRF by adding --use_char_representation=False or --use_crf=False
