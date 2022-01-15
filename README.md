<h1>Generative Autoencoder</h1>

## Label 포함여부 차이  

VAE에서 Label을 encoder와 decoder에 포함하여 넣어주면 좀 더 빨리 수렴함.  

|Epochs|VAE|Supervised Conditional VAE|
|--|--|--|
|1|![VAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/VAE_MNIST_1epoch.png)|![SCVAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/SCVAE_MNIST_1epoch.png)|
|5|![VAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/VAE_MNIST_5epoch.png)|![SCVAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/SCVAE_MNIST_5epoch.png)|
|10|![VAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/VAE_MNIST_10epoch.png)|![SCVAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/SCVAE_MNIST_10epoch.png)|

## Unsupervised Conditional VAE M2/M3 비교  

M2는 X로 y를 예측하는 분류기를 추가한 것이고, M3는 훈련된 M1 위에 y를 예측하는 분류기를 추가한 것임.  

|M2|M3|
|--|--|
|![M2](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/CVAEM2_MNIST.png)|![M3](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/CVAEM3_MNIST.png)|

M2는 400 epochs, M3는 M1은 200 epochs 훈련하고 분류기를 추가한 뒤 200 epochs 더 훈련했음. y의 차원은 MNIST의 Category와 동일하게 10개로 설정했으며 **각 row가 1개의 category를 나타냄.** 보이는 것 처럼 M3가 reconstruction을 좀 더 컨트롤하기 쉽게 잘 나눠진 것을 볼 수 있음. M2는 분리는 되어있는데 reconstruction이 잘 안됨. 하지만 M3도 Unsupervised로 분류했기에 원래 의도한 것 처럼 완벽하게 분리되지 않음.  

## DFCVAE [[Code]](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/dfcvae.py) [[Paper(arXiv)]](https://arxiv.org/abs/1610.00291)  

특징 : VAE는 Pixel-wise loss를 쓰기에 blurry함. 그래서 Pixel별로 reconstruction loss를 계산하는 대신 X와 Recon.을 CNN에 태운 다음 그 값을 비교해서 reconstruction loss 계산.  

|VAE|DFCVAE123|DFCVAE345|
|--|--|--|
|![VAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/VAE_celebA.png)|![DFC123](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/DFCVAE123_celebA.png)|![DFC345](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/DFCVAE345_celebA.png)|  

재밌는건 VGG의 저층을 사용하는 DFC123이 DFC345보다 generation시 좀 더 이쁘게 찍힌다는 것임. 논문에서의 예시도 Reconstruction시 DFCVAE345가 좀더 잘 찍혔지만 normal distribution으로 generation할 땐 DFCVAE123이 더 잘찍혔음  


## CAAE 

![Generation](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/caae_generation.png)  

남자 / 남자,콧수염 / 남자,콧수염,대머리 / 남자,콧수염,대머리,어림 / 남자,콧수염,대머리,어림,웃음 / 여자 / 여자,어림 / 여자,콧수염 / 여자,웃음 / 여자,어림,웃음  

[여자,콧수염]같이 데이터에 거의 없는 요소로 생성된 사진은 이상하게 나오는 것을 볼 수 있음. 남/여에 따라 머리카락 범위가 변하기도하고 대머리를 넣어주면 m자탈모같이 머리가 살짝 사라지는 모습에서 굉장히 학습을 잘 했다 느껴짐.(cycle gan은 형태를 바꾸는것에 굉장히 약했었음)  

