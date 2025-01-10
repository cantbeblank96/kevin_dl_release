# summary

## detect()

检出图片中的人脸框和关键点。使用 MTCNN。

### 使用示例

|  | 原图 | 带关键点和人脸框 |
| --- | --- | --- |
| image | ![image_0.jpg](plots/detect/image_0.jpg) | ![image_1.jpg](plots/detect/image_1.jpg) |


## align_and_crop()

根据关键点对人脸进行转正，并按照指定的人脸位置和画幅大小进行裁切。

### 使用示例

|  | 原图 | 带关键点和人脸框 | align_and_crop | using padding_ls 01 | using padding_ls 02 | using desired_size |
| --- | --- | --- | --- | --- | --- | --- |
| image | ![image_0.jpg](plots/align_and_crop/image_0.jpg) | ![image_1.jpg](plots/align_and_crop/image_1.jpg) | ![image_2.jpg](plots/align_and_crop/image_2.jpg) | ![image_3.jpg](plots/align_and_crop/image_3.jpg) | ![image_4.jpg](plots/align_and_crop/image_4.jpg) | ![image_5.jpg](plots/align_and_crop/image_5.jpg) |
| paras | / | / | {'face_size': 224} | {'face_size': 112, 'padding_ls': (0, -8, 0, -8)} | {'face_size': 112, 'padding_ls': (15, 0, 10, -8)} | {'face_size': 150, 'desired_size': 224} |
| image_shape | (344, 412, 3) | (344, 412, 3) | (224, 224, 3) | (112, 96, 3) | (137, 104, 3) | (224, 224, 3) |


注意，为了更好地展示转正、裁剪前后的人脸相对位置，因此在 align_and_crop 中使用的是"带关键点和人脸框"的图片作为输入。

## detect() align_and_crop() 串联

### 不同head pose下的人脸检测校正示例

人头模型数据来自：Head scan 13 (photogrammetry) by yaro.pro on Sketchfab，遵守 CC BY 4.0 DEED 协议。

<div class="sketchfab-embed-wrapper"> <iframe title="Head scan 13 (photogrammetry)" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/5e6d2804405449e6b3bd96cd12d8b1ab/embed"> </iframe> <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;"> <a href="https://sketchfab.com/3d-models/head-scan-13-photogrammetry-5e6d2804405449e6b3bd96cd12d8b1ab?utm_medium=embed&utm_campaign=share-popup&utm_content=5e6d2804405449e6b3bd96cd12d8b1ab" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> Head scan 13 (photogrammetry) </a> by <a href="https://sketchfab.com/yaro.pro?utm_medium=embed&utm_campaign=share-popup&utm_content=5e6d2804405449e6b3bd96cd12d8b1ab" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> yaro.pro </a> on <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=5e6d2804405449e6b3bd96cd12d8b1ab" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a></p></div>

检测 detect() 所用的参数为：

```
{
    "thresholds": [
        0.4,
        0.6,
        0.6
    ]
}
```

人脸转正 align_and_crop() 所用的参数为：

```
{
    "face_size": 200,
    "desired_size": 256
}
```

#### pitch

| pose | -60 | -30 | 0 | 30 | 60 |
| --- | --- | --- | --- | --- | --- |
| raw_image | ![raw_image_0.jpg](plots/detect_and_align_and_crop/pitch/raw_image_0.jpg) | ![raw_image_1.jpg](plots/detect_and_align_and_crop/pitch/raw_image_1.jpg) | ![raw_image_2.jpg](plots/detect_and_align_and_crop/pitch/raw_image_2.jpg) | ![raw_image_3.jpg](plots/detect_and_align_and_crop/pitch/raw_image_3.jpg) | ![raw_image_4.jpg](plots/detect_and_align_and_crop/pitch/raw_image_4.jpg) |
| detect_image | / | ![detect_image_1.jpg](plots/detect_and_align_and_crop/pitch/detect_image_1.jpg) | ![detect_image_2.jpg](plots/detect_and_align_and_crop/pitch/detect_image_2.jpg) | ![detect_image_3.jpg](plots/detect_and_align_and_crop/pitch/detect_image_3.jpg) | / |
| detect_res | None | [{'bbox': [250.62155151367188, 267.8813781738281, 630.9801635742188, 711.1162109375], 'landmarks': [[360.83612060546875, 469.09808349609375], [543.1746826171875, 471.18121337890625], [444.28326416015625, 602.9383544921875], [366.3970947265625, 609.438720703125], [522.4087524414062, 612.627685546875]], 'score': 0.9999855756759644}] | [{'bbox': [207.6573486328125, 173.43716430664062, 573.7564086914062, 658.10302734375], 'landmarks': [[324.04022216796875, 364.5065002441406], [486.5561218261719, 363.8453369140625], [407.4049987792969, 463.06463623046875], [330.71868896484375, 533.4786987304688], [485.2341003417969, 536.2158203125]], 'score': 1.0}] | [{'bbox': [257.5014953613281, 37.141666412353516, 605.9730834960938, 439.7900390625], 'landmarks': [[366.5904541015625, 185.39981079101562], [520.030029296875, 182.63783264160156], [447.1324768066406, 228.2477569580078], [365.45758056640625, 334.1454772949219], [517.9027099609375, 331.77850341796875]], 'score': 0.9999920129776001}] | None |
| align_image | / | ![align_image_1.jpg](plots/detect_and_align_and_crop/pitch/align_image_1.jpg) | ![align_image_2.jpg](plots/detect_and_align_and_crop/pitch/align_image_2.jpg) | ![align_image_3.jpg](plots/detect_and_align_and_crop/pitch/align_image_3.jpg) | / |


#### roll

| pose | -150 | -120 | -90 | -30 | -15 | 0 | 15 | 30 | 90 | 120 | 150 | 180 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_image | ![raw_image_0.jpg](plots/detect_and_align_and_crop/roll/raw_image_0.jpg) | ![raw_image_1.jpg](plots/detect_and_align_and_crop/roll/raw_image_1.jpg) | ![raw_image_2.jpg](plots/detect_and_align_and_crop/roll/raw_image_2.jpg) | ![raw_image_3.jpg](plots/detect_and_align_and_crop/roll/raw_image_3.jpg) | ![raw_image_4.jpg](plots/detect_and_align_and_crop/roll/raw_image_4.jpg) | ![raw_image_5.jpg](plots/detect_and_align_and_crop/roll/raw_image_5.jpg) | ![raw_image_6.jpg](plots/detect_and_align_and_crop/roll/raw_image_6.jpg) | ![raw_image_7.jpg](plots/detect_and_align_and_crop/roll/raw_image_7.jpg) | ![raw_image_8.jpg](plots/detect_and_align_and_crop/roll/raw_image_8.jpg) | ![raw_image_9.jpg](plots/detect_and_align_and_crop/roll/raw_image_9.jpg) | ![raw_image_10.jpg](plots/detect_and_align_and_crop/roll/raw_image_10.jpg) | ![raw_image_11.jpg](plots/detect_and_align_and_crop/roll/raw_image_11.jpg) |
| detect_image | ![detect_image_0.jpg](plots/detect_and_align_and_crop/roll/detect_image_0.jpg) | / | / | ![detect_image_3.jpg](plots/detect_and_align_and_crop/roll/detect_image_3.jpg) | ![detect_image_4.jpg](plots/detect_and_align_and_crop/roll/detect_image_4.jpg) | ![detect_image_5.jpg](plots/detect_and_align_and_crop/roll/detect_image_5.jpg) | ![detect_image_6.jpg](plots/detect_and_align_and_crop/roll/detect_image_6.jpg) | ![detect_image_7.jpg](plots/detect_and_align_and_crop/roll/detect_image_7.jpg) | ![detect_image_8.jpg](plots/detect_and_align_and_crop/roll/detect_image_8.jpg) | ![detect_image_9.jpg](plots/detect_and_align_and_crop/roll/detect_image_9.jpg) | ![detect_image_10.jpg](plots/detect_and_align_and_crop/roll/detect_image_10.jpg) | / |
| detect_res | [{'bbox': [364.4564514160156, 305.30279541015625, 743.6248779296875, 736.9730834960938], 'landmarks': [[477.6927490234375, 507.36407470703125], [592.1654052734375, 462.001220703125], [552.1463623046875, 541.3966674804688], [525.5625610351562, 621.3319091796875], [634.7294311523438, 578.88427734375]], 'score': 0.9259271025657654}] | None | None | [{'bbox': [335.9417419433594, 312.9685363769531, 712.1754150390625, 774.203369140625], 'landmarks': [[476.9778137207031, 472.5284423828125], [626.8736572265625, 561.8377685546875], [514.565673828125, 615.023193359375], [397.39447021484375, 619.0408325195312], [541.133544921875, 699.40478515625]], 'score': 0.9993402361869812}] | [{'bbox': [306.1201477050781, 257.1661376953125, 661.8829345703125, 727.568115234375], 'landmarks': [[417.762451171875, 433.4466552734375], [577.3548583984375, 478.6429138183594], [474.2560729980469, 552.162841796875], [375.5601806640625, 593.4993896484375], [532.6273803710938, 639.4456787109375]], 'score': 0.9999986886978149}] | [{'bbox': [207.6573486328125, 173.43716430664062, 573.7564086914062, 658.10302734375], 'landmarks': [[324.04022216796875, 364.5065002441406], [486.5561218261719, 363.8453369140625], [407.4049987792969, 463.06463623046875], [330.71868896484375, 533.4786987304688], [485.2341003417969, 536.2158203125]], 'score': 1.0}] | [{'bbox': [297.4172668457031, 276.3334045410156, 671.4969482421875, 746.4334106445312], 'landmarks': [[407.68182373046875, 486.78863525390625], [570.189208984375, 443.1029968261719], [510.79339599609375, 564.2474365234375], [459.9791259765625, 653.3760986328125], [606.6703491210938, 612.0220947265625]], 'score': 0.9999860525131226}] | [{'bbox': [338.7366027832031, 338.6726379394531, 749.2334594726562, 809.8982543945312], 'landmarks': [[460.12030029296875, 576.8323364257812], [611.5447998046875, 496.9690246582031], [568.5191650390625, 630.0532836914062], [540.7578125, 729.7971801757812], [686.4916381835938, 654.582275390625]], 'score': 0.9999574422836304}] | [{'bbox': [291.82708740234375, 240.31658935546875, 651.2684326171875, 585.6199340820312], 'landmarks': [[436.50372314453125, 406.6761474609375], [523.0560302734375, 365.92877197265625], [495.4193115234375, 451.328369140625], [473.27117919921875, 486.29730224609375], [550.7410888671875, 448.9317626953125]], 'score': 0.8008064031600952}] | [{'bbox': [411.8309631347656, 377.8095703125, 750.5617065429688, 741.895263671875], 'landmarks': [[556.27880859375, 518.651611328125], [643.4457397460938, 538.8365478515625], [583.372314453125, 581.4603271484375], [543.7402954101562, 619.145751953125], [612.18115234375, 634.5833740234375]], 'score': 0.9765224456787109}] | [{'bbox': [340.30609130859375, 288.4888916015625, 777.578125, 799.522216796875], 'landmarks': [[557.4313354492188, 475.4339599609375], [680.2457275390625, 573.1073608398438], [581.2861328125, 581.99853515625], [445.78082275390625, 599.8511962890625], [563.1920166015625, 683.4737548828125]], 'score': 0.9413009881973267}] | None |
| align_image | ![align_image_0.jpg](plots/detect_and_align_and_crop/roll/align_image_0.jpg) | / | / | ![align_image_3.jpg](plots/detect_and_align_and_crop/roll/align_image_3.jpg) | ![align_image_4.jpg](plots/detect_and_align_and_crop/roll/align_image_4.jpg) | ![align_image_5.jpg](plots/detect_and_align_and_crop/roll/align_image_5.jpg) | ![align_image_6.jpg](plots/detect_and_align_and_crop/roll/align_image_6.jpg) | ![align_image_7.jpg](plots/detect_and_align_and_crop/roll/align_image_7.jpg) | ![align_image_8.jpg](plots/detect_and_align_and_crop/roll/align_image_8.jpg) | ![align_image_9.jpg](plots/detect_and_align_and_crop/roll/align_image_9.jpg) | ![align_image_10.jpg](plots/detect_and_align_and_crop/roll/align_image_10.jpg) | / |


#### yaw

| pose | -90 | -60 | -30 | 0 | 30 | 60 | 90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_image | ![raw_image_0.jpg](plots/detect_and_align_and_crop/yaw/raw_image_0.jpg) | ![raw_image_1.jpg](plots/detect_and_align_and_crop/yaw/raw_image_1.jpg) | ![raw_image_2.jpg](plots/detect_and_align_and_crop/yaw/raw_image_2.jpg) | ![raw_image_3.jpg](plots/detect_and_align_and_crop/yaw/raw_image_3.jpg) | ![raw_image_4.jpg](plots/detect_and_align_and_crop/yaw/raw_image_4.jpg) | ![raw_image_5.jpg](plots/detect_and_align_and_crop/yaw/raw_image_5.jpg) | ![raw_image_6.jpg](plots/detect_and_align_and_crop/yaw/raw_image_6.jpg) |
| detect_image | ![detect_image_0.jpg](plots/detect_and_align_and_crop/yaw/detect_image_0.jpg) | ![detect_image_1.jpg](plots/detect_and_align_and_crop/yaw/detect_image_1.jpg) | ![detect_image_2.jpg](plots/detect_and_align_and_crop/yaw/detect_image_2.jpg) | ![detect_image_3.jpg](plots/detect_and_align_and_crop/yaw/detect_image_3.jpg) | ![detect_image_4.jpg](plots/detect_and_align_and_crop/yaw/detect_image_4.jpg) | ![detect_image_5.jpg](plots/detect_and_align_and_crop/yaw/detect_image_5.jpg) | ![detect_image_6.jpg](plots/detect_and_align_and_crop/yaw/detect_image_6.jpg) |
| detect_res | [{'bbox': [496.5079650878906, 146.81402587890625, 736.3521118164062, 557.741455078125], 'landmarks': [[676.2125244140625, 303.91668701171875], [708.61376953125, 286.8040771484375], [764.3544921875, 358.3699951171875], [700.2597045898438, 450.2082214355469], [725.2314453125, 436.7627258300781]], 'score': 0.999972939491272}, {'bbox': [401.49322509765625, 305.754638671875, 527.3079833984375, 454.9803466796875], 'landmarks': [[438.0669860839844, 363.3542175292969], [479.1484375, 360.2784423828125], [452.8587646484375, 388.25732421875], [448.7643127441406, 409.1600036621094], [483.1414489746094, 408.85546875]], 'score': 0.8082259893417358}] | [{'bbox': [356.6125183105469, 142.52349853515625, 678.3504028320312, 578.1112670898438], 'landmarks': [[560.0838012695312, 310.86627197265625], [669.3888549804688, 298.300048828125], [681.92138671875, 383.80474853515625], [583.2655029296875, 481.6845397949219], [671.0933837890625, 471.59552001953125]], 'score': 0.999996542930603}] | [{'bbox': [354.08062744140625, 130.1438751220703, 708.6588745117188, 599.8508911132812], 'landmarks': [[514.1168212890625, 321.61932373046875], [669.130859375, 316.2099609375], [632.730712890625, 414.28363037109375], [533.520751953125, 500.46929931640625], [663.9256591796875, 491.152099609375]], 'score': 0.9999830722808838}] | [{'bbox': [207.6573486328125, 173.43716430664062, 573.7564086914062, 658.10302734375], 'landmarks': [[324.04022216796875, 364.5065002441406], [486.5561218261719, 363.8453369140625], [407.4049987792969, 463.06463623046875], [330.71868896484375, 533.4786987304688], [485.2341003417969, 536.2158203125]], 'score': 1.0}] | [{'bbox': [242.96896362304688, 97.15921783447266, 591.6715087890625, 576.9723510742188], 'landmarks': [[299.6816101074219, 290.1266784667969], [451.94964599609375, 294.2207336425781], [332.24658203125, 392.8564147949219], [297.5914306640625, 465.3005065917969], [437.1089172363281, 469.333984375]], 'score': 0.9999747276306152}] | [{'bbox': [116.11695098876953, 104.15735626220703, 437.33477783203125, 556.6654663085938], 'landmarks': [[156.80218505859375, 288.58807373046875], [253.55169677734375, 288.39007568359375], [149.52662658691406, 376.3365478515625], [160.70208740234375, 469.89605712890625], [237.33262634277344, 463.2928466796875]], 'score': 0.9999940395355225}] | [{'bbox': [127.37367248535156, 132.0696258544922, 399.3570556640625, 545.940673828125], 'landmarks': [[178.5, 283.25067138671875], [242.0547332763672, 285.1585998535156], [157.8037109375, 364.342041015625], [181.73837280273438, 438.26605224609375], [234.5187530517578, 438.14434814453125]], 'score': 0.9999955892562866}] |
| align_image | ![align_image_0.jpg](plots/detect_and_align_and_crop/yaw/align_image_0.jpg) | ![align_image_1.jpg](plots/detect_and_align_and_crop/yaw/align_image_1.jpg) | ![align_image_2.jpg](plots/detect_and_align_and_crop/yaw/align_image_2.jpg) | ![align_image_3.jpg](plots/detect_and_align_and_crop/yaw/align_image_3.jpg) | ![align_image_4.jpg](plots/detect_and_align_and_crop/yaw/align_image_4.jpg) | ![align_image_5.jpg](plots/detect_and_align_and_crop/yaw/align_image_5.jpg) | ![align_image_6.jpg](plots/detect_and_align_and_crop/yaw/align_image_6.jpg) |


