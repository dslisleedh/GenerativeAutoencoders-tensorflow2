<h1>Generative Autoencoder 모델들을 구현해봄.</h1>

## DFCVAE [[Code]](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/dfcvae.py) [[Paper(arXiv)]](https://arxiv.org/abs/1610.00291)  

특징 : VAE는 Pixel-wise loss를 쓰기에 blurry함. 그래서 Pixel별로 reconstruction loss를 계산하는 대신 X와 Recon.을 CNN에 태운 다음 그 값을 비교해서 reconstruction loss 계산.  

|VAE|DFCVAE123|DFCVAE345|
|--|--|--|
|![VAE](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/VAE_celebA.png)|![DFC123](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/DFCVAE123_celebA.png)|![DFC345](https://github.com/dslisleedh/GenerativeAutoencoders-tensorflow2/blob/main/imgs/DFCVAE345_celebA.png)|  

재밌는건 VGG의 저층을 사용하는 DFC123이 DFC345보다 generation시 좀 더 이쁘게 찍힌다는 것임. 논문에서의 예시도 Reconstruction시 DFCVAE345가 좀더 잘 찍혔지만 normal distribution으로 generation할 땐 DFCVAE123이 더 잘찍혔음  
