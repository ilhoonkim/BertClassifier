# BertClassifier
Google에서 제공하는 NLP 모델인 BERT를 통해 한글어 자연어 처리를 시도하는 과정에서 더 나은 방식을 찾고 정리하고 있습니다. 해당 부분은 전이학습(Fine-tunning) 과정에서 생각한 점과 보완한 부분을 정리하였습니다.

# 전이학습 TASK
BERT는 트랜스포머로 인코딩하여 학습한 사전학습 모델이라고 생각합니다.   
전이학습의 경우는 완성된 사전학습 모델 위에 FC layer를 한층 더 만들어 텍스트를 넣고 결과값을 예측하게 합니다.   
input으로 넣는 텍스트의 양식이나 혹은 예측하려는 값에 따라서 BERT Fine-tunning의 TASK가 달라지는 것입니다.  



**BERT FINE TUNNING TASK**
![그림4-bert-experiment-result](https://user-images.githubusercontent.com/45644085/84091910-d6f8c080-aa30-11ea-8098-a7c9a598d79f.png)

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

BERT Fine-Tunning의 TASK는 목적에 따라 다음과 같이 존재합니다.   
전이학습은 전이학습하려는 목적에 맞게 데이터의 형태만 지정해주면 된다고 보시면 됩니다.   
두 문장의 감정이 같은지를 판단하는 QQP를 수행하기 위해서는 문장 쌍과 해당 문장의 감정이 같은지 여부에 대한 라벨만 있다면 학습이 가능할 것입니다. 영화 리뷰의 감성을 분석하는 STT-2의 경우는 단일 문장과 해당 문장이 긍정인지 부정인지의 라벨만 존재하면 됩니다.   

# 전이학습 하기
**run_classifier.py** 참조   
잘 생각해보면 TASK는 이미 정해진 데이터로 그러한 학습을 수행하였을 뿐이고 저는 단일 문장이 들어오면 해당 문장의 의도를 구분하고 싶었습니다.
그래서 SST-2의 라벨을 바이너리가 아니라 여러개로 하여 학습해보기로 했습니다.



