# BertClassifier
Google에서 제공하는 NLP 모델인 BERT를 통해 한글어 자연어 처리를 시도하는 과정에서 더 나은 방식을 찾고 정리하고 있습니다. 해당 부분은 전이학습(Fine-tunning) 과정에서 생각한 점과 보완한 부분을 정리하였습니다.

# 전이학습 하기
BERT는 트랜스포머로 인코딩하여 학습한 사전학습 모델이라고 생각합니다.   
전이학습의 경우는 완성된 사전학습 모델 위에 FC layer를 한층 더 만들어 텍스트를 넣고 결과값을 예측하게 합니다.   
input으로 넣는 텍스트의 양식이나 혹은 예측하려는 값에 따라서 BERT Fine-tunning의 TASK가 달라지는 것입니다.   

**BERT FINE TUNNING TASK**
|TASK명|수행|
|---|:---:|
| **MNLI**  |두개의 문장을 주고 두문장이 같은 의미인지 모순인지 무관한지 판단|
| **STS-B**  |두 문장의 의미 유사도를 1-5점으로 판단|
| **MRPC**  |두 질문이 같은 의미의 질문인지 판단|
| **QQP**  |두 문장의 감정이 같은지를 판단|
| **SWAG**  |다음에 올 문장으로 알맞은 것은?(4지 선다)|
| **SST-2** |영화 리뷰에서 감정 분석(binary)|
| **CoLA**  |문장의 문법이 맞는지 판단|
| **SQuAD**  |질문을 보고 지문에서 정답 span을 찾기|
| **CoNLL**  |단어의 entity 종류를 판단(Person, Location 등등..)|


|   |Hidden_layer|Hidden_size|attention_head|
|---|:---:|:---:|:---:|
| **tiny**  |2|128|8|
| **small**  |4|512|8|
| **medium**  |8|512|8|
| **base**  |12|768|12|
| **large** |24|1024|16|
