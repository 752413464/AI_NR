# AI_NR
note code of studying Ai_nr 

1 Data Driven

噪声数据主要分为：pg-model，real-model， gan-model，awgn-model
a.首先awgn-model就图一乐，之前在rgb域有人加噪声就是用的这种，但问题是到rgb域噪声早都不是线性了，所以还是老老实实转raw。
b.gan-model看趋势是未来噪声合成的一个点，但ELD和rethink synth等paper点名了标定好的pg远强于但当前阶段的gan-mode，非极端条件下的pg-model是够用的，除非极端暗光等考虑ELD的加row-noise，后话不表。
c.除了real-model基本就只能考虑标定senor合成pg-model，目前也是这样考虑的，clean数据现在有zte和旷视的去噪比赛，以及SID的两百张大图，可以考虑ntire2019旷视的操作，加点bayer-aug，数据暂且这么处理。
d.理论上直接拟合real-model如sidd等提供的data最合适，但前期由于种种原因没有用好，可能是model确实太小，盲拟合效果差强人意，但有更好的噪声数据解决方案，最后再考虑它。


2.NN model
a.端侧AI芯片的算力大致在1-5T，直接排除一些堆算力的model如MPR和Restformer等，后续加工时可以考虑对这些model裁剪。
b.FLops->param, MPR等param倒不算很高，但Flops过高，处理单帧已经很耗时，基本不可能实时，所以设计model的基本原则是param高且Flops尽可能低，模型表征能力强，且推理速度快。
c.搞个高Flops低param的model有手就行，旷视5月份的meg cup前三名让我学到了，直出式model直接狂卷roll，能重复多少次重复多次，Flops已经上天了，params还在0.1M左右徘徊，确实屌，不太清楚旷视为什么要限制param却不关注Flops，按理说这样的model没有任何意义，部署效果我推测不会很好，知乎有个老哥试过前十名的model，基本全部拉跨，对应了我对model设计的想法。。
c.PMRID之类的商用算法已经给出成熟解决方案，U-net-like架构四次下采样，融合short-cut和long-skip，再用consepth拉低params，基本直接包圆low level task，作为入手借鉴的model，修改时主要对激活做些调整。
d.model可选择的太多了，ntire 22的efficent sr 赛道上的model基本都可用，那边毕竟评价标准是综合性能和效率，因此可用性我认为强于meg cup，需要时可以参考前几名的model。

3.Train categoires
a.patch_size，大图就全部切小块一个个喂model，不要浪费，前期操作时因为data浪费太严重，model性能断崖式下跌，引以为戒。
b.lr的初始1e-4和1e-3还是有区别，cosine衰减策略就先用着，暂时没啥问题，warm-up能开就开，最后adam就不改了。
c.loss function本应是最重要的一环，当前low level loss function 主要是L1, L2，L1+e等pixel-wise loss，也可以辅助mssim和sobel等loss，但半个月的实验发现，性能不稳定，权当这部分的尝试。
d.loss部分的补充，旷视meg cup的评价指标是(1/gt)**0.5作为权重，乘给L1作评价，不太清楚这样设置的意义，个人认为本质是还是L1。但是明白了打比赛就要按照评分规则设计loss，前几名都是这样，十名开外的都是L1练。

4.others
a.相同量级的mode，理论上盲去噪性能不如非盲，考虑盲不盲可以先从数据入手，比如用好标定的参数作VST变化（PMRID），或者生成noise-map直接concat进model作训练（FFDNet家族
b.如果能进行非盲尽可能非盲，但是计算代价和constrain较多，目前这块还在思考和实验。
c.手头有四五个camera的noise profile，在思考到底给个range随机合成噪声，还是按照标定好的参数一一合成，个人思考随机合成的泛化性要强一些，虽然有可能合成出camera拍不出来的noise，后续做对比后可以给出结论。
