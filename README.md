# BertClassifier
Google에서 제공하는 NLP 모델인 BERT를 통해 한글어 자연어 처리를 시도하는 과정에서 더 나은 방식을 찾고 정리하고 있습니다. 해당 부분은 전이학습(Fine-tunning) 과정에서 생각한 점과 보완한 부분을 정리하였습니다.

# 전이학습 TASK
BERT는 트랜스포머로 인코딩하여 학습한 사전학습 모델이라고 생각합니다.   
전이학습의 경우는 완성된 사전학습 모델 위에 FC layer를 한층 더 만들어 텍스트를 넣고 결과값을 예측하게 합니다.   
input으로 넣는 텍스트의 양식이나 혹은 예측하려는 값에 따라서 BERT Fine-tunning의 TASK가 달라지는 것입니다.  
![파인튜닝 원리](https://user-images.githubusercontent.com/45644085/84963416-4878f300-b144-11ea-8b47-abfb0930a89d.JPG)   

버트 모델은 파인튜닝 시에 FC(fully connected) 레이어를 하나 추가해주는 간단한 방식을 취하고 있습니다.


**BERT FINE TUNNING TASK**
![그림4-bert-experiment-result](https://user-images.githubusercontent.com/45644085/84091910-d6f8c080-aa30-11ea-8098-a7c9a598d79f.png)

|TASK명|수행|
|---|:---:|
| *MNLI*  |두개의 문장을 주고 두문장이 같은 의미인지 모순인지 무관한지 판단|
| *STS-B*  |두 문장의 의미 유사도를 1-5점으로 판단|
| *MRPC*  |두 질문이 같은 의미의 질문인지 판단|
| *QQP* |두 문장의 감정이 같은지를 판단|
| *SWAG*  |다음에 올 문장으로 알맞은 것은?(4지 선다)|
| *SST-2* |영화 리뷰에서 감정 분석(binary)|
| *CoLA* |문장의 문법이 맞는지 판단|
| *SQuAD* |질문을 보고 지문에서 정답 span을 찾기|
| *CoNLL* |단어의 entity 종류를 판단(Person, Location 등등..)|

BERT Fine-Tunning의 TASK는 목적에 따라 다음과 같이 존재합니다.   
전이학습은 전이학습하려는 목적에 맞게 데이터의 형태만 지정해주면 된다고 보시면 됩니다.   
두 문장의 감정이 같은지를 판단하는 QQP를 수행하기 위해서는 문장 쌍과 해당 문장의 감정이 같은지 여부에 대한 라벨만 있다면 학습이 가능할 것입니다. 영화 리뷰의 감성을 분석하는 STT-2의 경우는 단일 문장과 해당 문장이 긍정인지 부정인지의 라벨만 존재하면 됩니다.   

# 전이학습 하기
**run_classifier.py** 참조   
잘 생각해보면 TASK는 이미 정해진 데이터로 그러한 학습을 수행하였을 뿐이고 저는 단일 문장이 들어오면 해당 문장의 의도를 구분하고 싶었습니다.
그래서 SST-2의 라벨을 바이너리가 아니라 여러개로 하여 학습해보기로 했습니다.

## 1.데이터 만들기
우선 기존에 nsmc 데이터에서 일부만 간단하게 분류해보고 싶었습니다. 바이너리가 아닌 분류에 의미를 두겠습니다.      
그래서 영화 리뷰 데이터를 어떻게 분류할까하다 리뷰 내용에 대해 분류해보기로 했습니다.   
nsmc 데이터를 한번 쭉 보니 크게 1.연기력 2.스토리/구성 3.메세지/의미 4.영상미 5.영화음악 6.역사/현실고증 정도로 리뷰를 분류할 수 있을 것 같았습니다. 그래서 해당 분류에 들어갈만한 키워드를 구성하여 정규식을 통해 검출하여 각각 리스트에 담아보았습니다.

```
import re
import pandas as pd

nsmc_data = pd.read_csv("nsmc_corpus.txt", sep="delimeter", header = None)
nsmc_data["review"] = nsmc_data

actor = re.compile('연기력')
story = re.compile('스토리|각본|구성|소재')
message = re.compile('메시지|메세지|의미')
sound = re.compile('음악|bgm')
scene = re.compile('영상미|촬영기법')
history = re.compile('고증|왜곡|역사적 사실|역사적 배경|사료')

연기력_list = []
스토리_list = []
메세지_list = []
영상미_list = []
음악_list = []
고증_list = []

for i in tqdm_notebook(nsmc_data["review"]):
    if actor.search(i):
        연기력_list.append(i)
    if story.search(i):
        스토리_list.append(i)
    if message.search(i):
        메세지_list.append(i)
    if scene.search(i):
        영상미_list.append(i)  
    if sound.search(i):
        음악_list.append(i)            
    if history.search(i):
        고증_list.append(i)  
```
다음과 같은 결과를 얻을 수 있습니다.
```
>>>print(고증_list[:20])

...
['영화가 끝난후 텅비어버린 가슴..진실이 왜곡되는 정치적 이념의 송가..', '사실 왜곡 한 스푼, 한쪽 시각 한 스푼', '역사왜곡은 뭐 각색의 영역이라 그렇다치고 영화자체로도 참 B급 스런 쌈마이영화.그냥 머리를 비우고 멍하니 시간을 때우고 싶은게 아니라면 추천하진 않습니다', '쓰레기. 살아있는 고증오류의 신화', '마지막 너무 먹먹하고 슬프네요 부모님 울때 ㅜㅜ 영상미가 신선하고 괜찮았지만 미야자키님이 은퇴작으로 이걸 했더라면;;; 왜 왜곡 영화 만들어서...; 막판에 욕을 먹는지;;; 쩝', '독실한 신자인 제 친구도 보고 별로라구 하네요. 종교적으로 왜곡된게 많다고... 전 종교 안믿어서 모르겠지만 영화 자체도 그리 긴장감 없고 좀 지루했습니다.', '그냥 영화판 자서전이네.. 여러 고증을 하려는 점은 훌륭하나 일반인이 이 영화를 볼 이유가 전혀 없고 결정적으로 추상 표현주의 작품에 대해 경멸을 갖게 됐다', '고증거친 화면과 배경음악, 탄탄한 시나리오로..', '묻지도 따지지도 않고, 일본 /2차세계대전/ 감동적인이야기 세가지 키워드만 봐도 각 나온다. 당시 상황에 비록침략국이었을지라도 어린아이들은 피해자였을수있다고좋게 생각하려해도 사과없이 오히려 역사를 왜곡하려는 그네들이 만든 영화라니 심히 의도가 의심된다', '오늘 다큐멘터리랑 비교해 보면서 진짜 실제 상황이랑 똑같아서 엄청 놀랬다.. 너무 사실적인 고증이 재미를 반감시키지만.. 북베트남군의 용기는 정말 대단했다.. 10중 8,9 죽을 줄 알면서도.. 우리는 저럴 수 있을까?', '역사부정과 역사왜곡 국가의 만남. 내용도 그다지...', "어느순간 그냥 '뭐지? 헐 역사적 배경은 당연히 알고 있지만 이건아니다", '정말 끔찍하게 암걸리는 영화였다제목을 병림픽이라고 하지그나마 대통령이랑 유해진역할이 암 안걸렸음그리고 주인공남자는 왜 안걸리냐 버프받았냐캐릭터 거의 전부가 다 극혐이였음 근데 그게고증이 잘되보이긴함 우리나라 영화라서', '이거이거, 왜곡이 너무 심하다?', '논란 여부를 떠나 영화 자체가 재미없음. 차라리 영화에서도 언급되었던 안창남이 영화로 제작되었다면 더 좋았을 것을. 객관적 사실에 바탕으로 한 것에 가미된 픽션이 너무나도 어처구니가 없음. 그로 인해 왜곡된 역사를 사실로 인식하는 사람들도 있음', '역사왜곡 정말 할말이없음', '공중파 방송국은 반성좀 하자. 이런게 드라마지. 막장, 역사왜곡 따위 그만하고...', '개연성없는 전개, 상황에 맞지않는 대사, 거기다 왜곡된 역사인식까지. 정말 엉망진창', '곰TV들어갓다 자동으로 재생되길레 봤는대 역사왜곡이 심각함.우리나라나 남에껄 자기들꺼라 홍보하네....황당무게하다 못해 역겨움....초반 15분만 약간 코믹하게 웃겼고 나머진 역사날조의 일본우익놈들 좋아라 하게 만든 병맛 영화.', '뭐 역사를 다룬영화니 역사왜곡이느니 어쩌구하는 인간들많다 근데 영화는 영화일뿐이고전쟁중에서의 시점은 스파르타시점이니 그 시점에서의 페르시아가 그렇게보인다는 식으로 꾸민건데(적군이고 스파르타 입장이니) 그냥 영화로봐라 영화로 무슨 역사공부하냐?']
```
역시나 각 리스트의 데이터 수가 차이가 있어 각각 200개씩만 준비했습니다.

```
>>> print(len(연기력_list))
>>> print(len(스토리_list))
>>> print(len(메세지_list))
>>> print(len(영상미_list))
>>> print(len(음악_list))
>>> print(len(고증_list))

...
1351
9692
1598
623
1816
234
```

이렇게 하여 최종 학습셋이 만들어졌습니다. 총 1200개의 데이터이며 6개의 분류 라벨을 가지며 각 라벨마다 200개씩 데이터가 준비되었습니다.
```
nsmc_total = []
for i in 연기력_list[:200]:
    nsmc_total.append(i)
for i in 스토리_list[:200]:
    nsmc_total.append(i)
for i in 메세지_list[:200]:
    nsmc_total.append(i)
for i in 영상미_list[:200]:
    nsmc_total.append(i)
for i in 음악_list[:200]:
    nsmc_total.append(i)
for i in 고증_list[:200]:
    nsmc_total.append(i)

label_list = []
for i in range(200):
    label_list.append('연기력')
for i in range(200):
    label_list.append('스토리/구성')
for i in range(200):
    label_list.append('메세지/의미')
for i in range(200):
    label_list.append('영상미')
for i in range(200):
    label_list.append('영화음악')
for i in range(200):
    label_list.append('역사/현실고증')


df = pd.DataFrame(data = nsmc_total, columns=["sentence"])
df['label'] = label_list
```
## 2.학습하기
보통 train 하기위해서는 검증셋(dev.tsv) 테스트셋(test.tsv)를 모두 준비해야 되나 그건 생략하고 일단 학습셋을 복사해서 명칭만 바꾸어서 사용하겠습니다. 
제 사전학습 모델을 공개하기는 어려워 우선 전이학습의 경우는 BERT-Base, Multilingual Cased 모델을 사용하였습니다.
```
  python run_classifier.py \
  --task_name=sst2 \
  --do_train=true \
  --do_eval=true \
  --data_dir=$BERT_BASE/data/nsmc \
  --vocab_file=$BERT_BASE/pretrained_models/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$BERT_BASE/pretrained_models/multi_cased_L-12_H-768_A-12/bert_config.json \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --init_checkpoint=$BERT_BASE/pretrained_models/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --num_train_epochs=10000.0 \
  --output_dir=$BERT_BASE/finetuning_models/nsmc_class_output

```
![클래시파이어학습](https://user-images.githubusercontent.com/45644085/84970043-df4dab80-b154-11ea-9859-bfe9d4a129f4.JPG)   
이렇게 학습이 진행됩니다. 

loss가 로깅되는 이유는 [BERT 사전학습 관련 깃헙](https://github.com/ilhoonkim/BertPre-training#%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5-%ED%95%98%EA%B8%B0)을 참조하시길 바랍니다.   
## 3. 예측하기(결과 확인하기)
run_classifier.py의 예측 부분은 test.tsv 파일을 읽어와서 각각 예측하여  정확도를 계산하는 것인데요.   
실제로 만들어진 모델을 즉각적으로 한 문장씩 테스트해보기 위해서 수정을 하였습니다.   
```
 if FLAGS.do_predict:
    while True:
        try:
            test_sentence = input('input sentence: ') # 쉘에서 바로 입력한 문장을 test_example로 만든다.
            predict_examples = []
            guid = "test-1"
            label = "연기력"
            text_a = tokenization.convert_to_unicode(str(test_sentence))
            predict_examples.append(
                  InputExample(guid=guid, text_a=text_a, label=label))

            num_actual_predict_examples = len(predict_examples)
            ....
            ...
            ..
            .
            result = estimator.predict(input_fn=predict_input_fn)

            for (i, prediction) in enumerate(result):
                label = prediction["probabilities"].tolist().index(max(prediction["probabilities"]))
                print(f'CATEGORY: {label_list[label]}    확률 :{prediction["probabilities"].tolist()[label]}') # 예측 결과를 바로 프린트해주기
        except Exception as e:
            print(f'{type(e).__name__}: {" ".join(list(e.args))}')

```
실제로 --do_predict=True로 주고 문장을 넣어서 학습한 모델을 테스트해 본 결과입니다.   
모델은  8만 스텝 정도 학습하였습니다.   

![프리딕트결과](https://user-images.githubusercontent.com/45644085/84995858-77fd1f00-b187-11ea-8753-3e56b50510d6.png)
