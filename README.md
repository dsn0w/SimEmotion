# SimEmotion

The main codes are shown in the main.py file. You can change different models and datasets.

### Requirements
Linux server with GPU. (Recommended：NVIDIA RTX 3090 and above.)

The requirements are the same as for CLIP (https://github.com/openai/CLIP).

### Data

The datasets and splits are referenced from the WSCNet(https://github.com/sherleens/WSCNet). 
Special thanks to Prof. Jufeng Yang and his team.

The results of the detection part of the model have been saved in a separate cate info folder for ease of processing. (BaiduNetDisk: password:[VipL](https://pan.baidu.com/s/1JSpkmoJQGviBKIPDwOQlIg))

### The data file directory structure：
cate info
- EmotionROI
  - anger1.txt
  - anger2.txt
    - ...
- FI
  - ...
- TwitterI
  - ...
- TwitterII
 -...

Emotion6
- train
  - anger
    - anger1.jpg
    - anger2.jpg
    - ...
  - disgust
  - ...
- test
  - ...

FI
- train
  - amusement
  - anger
  - ...
- val
  - ...
- test
  - ...

Emotion6_2
- train
  - negative
    - ...
  - positive
    - ...
- test
  - ...

FI_2
- train
  - negative
    - ...
  - positive
    - ...
- val
  - ...
- test
  - ...

TwitterI
- total
  - 1
    - train
      - negative
        - ...
      - positive
        - ...
    - test
        - ...

TwitterII
- train
  - negative
    - ...
  - positive
    - ...
- test
  - ...


More information to be updated.

------------

## References
```
[1] Deng S, Shi G, Wu L, et al. Simemotion: A simple knowledgeable prompt tuning method for image emotion classification[C]//Database Systems for Advanced Applications: 27th International Conference, DASFAA 2022, Virtual Event, April 11–14, 2022, Proceedings, Part III. Cham: Springer International Publishing, 2022: 222-229.

[2] Deng S, Wu L, Shi G, et al. Simple but Powerful, a Language-Supervised Method for Image Emotion Classification[J]. IEEE Transactions on Affective Computing, 2022.(Early Access)
```
