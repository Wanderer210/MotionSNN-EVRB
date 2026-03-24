1. Gopro测试
没有保存图像：works和batchsize都为2，1个GPU
(mosnn) zy@w680:~/data/zy/zhaoyue/MotionSNN-main$ python MOSNN_GoPro_test.py
total params is 7.1137 MB
local rank 0 begin to validate...
Test GoPro: PSNR: 35.1756, SSIM: 0.9545

没有保存图像：works和batchsize都为2，2个GPU
Test GoPro: PSNR: 35.2914, SSIM: 0.9547

batchsize都为6，works为4
Test GoPro: PSNR: 35.2914, SSIM: 0.9547
python MOSNN_GoPro_test_ssim.py

Test GoPro: PSNR: 35.2914, SSIM: 0.9766

2. REBlur微调训练并测试
python MOSNN_REBlur_funtine_train.py \
  --pretrained-path /home/zy/data/zy/zhaoyue/MotionSNN-main/MOSNN_out/best_35.6560_0.9780_7232_gopro_T10_model_distribute.pth 




final_best_psnr_36.5527_ssim_0.9830_epoch_0030_reblur_T10_model_distribute.pth

python MOSNN_REBlur_test.py --load /home/zy/data/zy/zhaoyue/MotionSNN-main/MOSNN_out_REBlur_LeakyRelu/MOSNN_REBlur_20260318_224033/checkpoints/final_best_psnr_36.5527_ssim_0.9830_epoch_0030_reblur_T10_model_distribute.pth --save-results True --results-dir results_REBlur


Test: PSNR=35.9600, SSIM=0.9819

3. 利用Gopro权重直接测试REBlur
python MOSNN_REBlur_test.py --load /home/zy/data/zy/zhaoyue/MotionSNN-main/MOSNN_out/best_35.6560_0.9780_7232_gopro_T10_model_distribute.pth --save-results True --results-dir results_REBlur

REBlur Test: PSNR=34.2406, SSIM=0.9740

4. 320*262  padding
python MOSNN_REBlur_test_padding.py --load /home/zy/data/zy/zhaoyue/MotionSNN-main/MOSNN_out/best_35.6560_0.9780_7232_gopro_T10_model_distribute.pth --save-results True --results-dir results_REBlur_padding




在图像质量评估中， SSIM（结构相似性）值在旧版和新版库之间存在差异 是一个非常普遍的现象。你发现旧版环境得到的 SSIM 值更高，主要有以下几个核心原因：

### 1. 权重分配逻辑的不同 (Gaussian Weights)
- 旧版环境 (0.18.3) ：在默认调用 compare_ssim(..., multichannel=True) 时，通常使用的是 均匀权重窗口（Uniform Window） 。这意味着在计算相似度时，滑动窗口内的每一个像素点对结果的影响是完全一样的。
- 新版/参考代码 (0.19+) ：在你提供的参考脚本 test1.py 中，显式开启了 gaussian_weights=True 和 sigma=1.5 。
  - 影响 ：高斯加权会让窗口 中心区域 的像素拥有更高的权重。由于图像的结构细节通常集中在局部中心，高斯加权对结构偏差更加敏感，计算出的结果往往比均匀权重更“严格”，因此数值会偏低。
### 2. 边界处理方式的变更 (Padding & Boundary)
- 随着 scikit-image 的更新，库对图像边缘（Border）的处理变得更加保守。
- 旧版本在处理边缘像素时，可能采用了更简单的填充方式，这在某些情况下会由于边缘效应的忽略而导致整体平均分略高。新版本则更倾向于严格遵循 Wang 等人原始论文中的标准实现，对边缘的处理更严谨，从而反映出更真实的（通常也更低的）结构相似度。
### 3. 多通道（Multichannel）处理的演进
- 旧版 ：通过 multichannel=True 简单地对 R、G、B 三个通道分别计算再取平均。
- 新版 ：引入了 channel_axis 并优化了内部的计算流。虽然逻辑相似，但在底层浮点数运算和中间变量的裁剪（Clipping）上做了优化。
- 差异 ：由于 SSIM 涉及多次平方和开方运算，微小的浮点数截断差异在累加后也会反映在最终的小数点后三、四位上。
### 4. 数据范围的隐式处理
- 在旧版库中，如果未显式指定 data_range ，库会根据图像的数据类型自动推断（例如 uint8 默认为 255）。
- 如果你在旧环境里没写 data_range=1.0 ，而图像又是 [0, 1] 范围的浮点数，库的推断逻辑可能与新版不同，导致计算公式中的常数项 [ o bj ec tO bj ec t ] C 1 ​ , C 2 ​ 比例失调，从而产生不正常的“高分”。
### 总结建议