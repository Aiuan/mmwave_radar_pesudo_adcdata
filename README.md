# mmwave radar pesudo adcdata

## case 1

![image-20230921161933533](./assets/image-20230921161933533.png)

当dopplerFFT环节开启hanning窗滤波，点云生成时，遗漏了最近的目标点

![image-20230921161814241](./assets/image-20230921161814241.png)

![image-20230921161853162](./assets/image-20230921161853162.png)

## case2

当dopplerFFT环节未开启hanning窗滤波，点云生成时，未遗漏目标点，但速度模糊现象愈发严重

![image-20230921162427303](./assets/image-20230921162427303.png)

![image-20230921162529472](./assets/image-20230921162529472.png)

![image-20230921162543040](./assets/image-20230921162543040.png)

## case3

当dopplerFFT环节未开启hanning窗滤波，开启速度扩展算法选项，点云生成时，未遗漏目标点，速度模糊现象有所缓解

![image-20230921162650557](./assets/image-20230921162650557.png)

![image-20230921162807134](./assets/image-20230921162807134.png)

