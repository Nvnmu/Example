# Example
从CLIP到cogvlm[https://www.zhihu.com/question/495007578/answer/3441793158]
clip [https://zhuanlan.zhihu.com/p/660476765]

# 为什么需要模态对齐

这篇文章会简略介绍模态对齐的本质和从CLIP到CogVLM的motivation、模型架构、损失函数、训练过程。在单模态大模型领域，为了用数字表达图像或文字，我们会采用embedding技术向量化我们的图像或文字，再将其输入到大模型内进行训练。最终，模型形成一套专属于自己训练数据集的embedding空间表示，就像生活在不同地区的人们有不同的语言，举个更详细的例子:各自的大模型有各自的用于分词和embedding的tokenizer，这就是模型的专属语言，归根结底，大模型学习的就是我们数据集的空间分布特征。而在多模态大模型中，我们的输入既有语言又有图像，我们该怎么让语言和图像具有同样的embedding空间表示呢？通俗地讲，使其具有同一种"语言"，不至于交流不通。举个例子,分别给模型的图像处理模块和语言处理模块输入一只小鸟的照片和自然语言描述"一张小鸟的照片",两个模块输出的embedding向量应该是相似的。这样，我们就可以把embedding后的图像转化到和文字embedding相同的分布空间，就能让我们的自然语言模型能够处理图像。注意，因为小鸟图片中的信息是多样的，这不仅是一只鸟，里面可能还包括鸟的种类、颜色以及大小等。所以我们会用“相似”来形容两个向量，而不是“相同“。这一点很重要，关系到后面模型的改进。我们还要思考，当我们分别有优秀的预训练视觉模型和语言模型时，如果我们按以上的方法，通过同时调整两种预训练模型的权重和embedding来实现模态对齐的时候，实际上是改变了预训练任务时的数据分布的，就是说"图片-文本对"的向量空间和预训练时的”文本“向量空间分布是不同的，预训练大模型只是学到了纯粹文本的分布，但没有学到”图片-文本“的分布，这会造成预训练任务的性能下降甚至是灾难性遗忘，我们该用什么方法避免这种衰退呢？有没有办法可以实现多模态对齐，而又不损失模型的原本性能呢？

# CLIP
Learning Transferable Visual Models From Natural Language Supervision
​arxiv.org/abs/2103.00020

## Motivation
相较于在自然语言处理领域取得成功的自监督学习，CV仍然在使用有监督学习，这使得模型的训练数据集很少，模型缺少泛化性，且人为标注数据集的价格昂贵。所以CLIP想从NLP领域借鉴这种自监督方法，用更大的数据集来训练更好的多模态模型，这种训练方式让模型的zero-shot能力更好，可以从下面的dataset构建中看到原因。

## Dateset
相较于人工定义数据集标签，CLIP选择直接先从wiki上爬取出现次数超过100的词汇，再用这些词汇构成query,去搜索图像，最后构成总计400million的image-query对。这里值得注意的是一张图片可能有多个query,比如对于同一张猫咪图片，其query可能是：小猫咪、室内宠物、橘色动物。这种构建方式相较于ImageNet的单标签来讲，模型能够捕捉到图片中的多种语义，大大提升了泛化性。哪怕训练时没有见过某个种类，也能又较好的效果，比如训练集中有蓝色的狗和绿色的猫，而预测的时候，如果图片里出现的是绿色的狗，这是从未出现的，但模型依然可以从颜色和类别出发，得出这是一只绿色狗狗。

## Architecture
![image](https://github.com/Nvnmu/Example/assets/93867177/c45c78be-4ab8-4fc7-ab66-7dc0ff8e98ea)
我们在开头讲过，如果用预训练好的模型直接去调整模型参数来做对齐，会有很大的数据分布偏移，使模型丧失原有性能。得益于巨大的数据集和丰富的计算资源，CLIP能从头开始训练Text Encoder和Image Encoder，其中前者为GPT-2架构，text的输出feature为语句末尾的特殊token[EOS]记作T，后者为ResNet或者ViT,输出为开头的特殊token [CLS]，记作I

## Training
Loss
训练过程采用了Image-Text Contrastive损失，简单来讲就是对比I和T的相似度，既要横向比较固定一个I,哪个T是最相似的。也要纵向比较，固定一个T，哪个I是最相似的。我们最大化正确I-T对的相似度，减少非匹配对的相似度，从而实现模态对齐。横向和纵向分别对比，对于每个图像，我们希望找到与之最匹配的文本描述；同样，对于每个文本描述，我们也希望找到与之最匹配的图像。如上图，同一个batch内，其他的image-text pair就是天然的负类，而只有对角线上的image-text pair才是正类。

# ALBEF
## Motivation
在之前的CLIP中，模型虽然由具有生成能力的GPT-2作为组成部分，但是CLIP本身却丧失了生成能力，不能由模型生成对应图片的caption,ALBEF想要想法子恢复一些生成能力。
CLIP的数据来自网络，数量庞大，但其中不乏有噪音，ALBEF希望能减轻这些噪音对模型训练的影响.
## Architechture
![image](https://github.com/Nvnmu/Example/assets/93867177/0602a890-79a6-493e-afe8-944097353816)
image encoder是基于imageNet-1k训练的vit
text encoder 和 multimodal encoder是来自于bert-base,从中间切开得来的并初始化的，但是multimodal encoder的cross attention需要自己初始化。
右侧的动量模型是左侧模型的参数的指数移动平均，可以帮助模型更平稳的训练并缓解脏数据的困扰
## Loss
### ITC
和CLIP一致，直接对比image encoder和text encoder的[cls]输出的相似度，除了本身输入的image-text对是正例以外，其他都是负例。那么这里就有一个潜在的问题，对于一个image来说，其可以有多个合理的text,如果我们只以image-text为正例，其他都为负，这是不是有点不公平呢？这里我们留个坑。此外我们在计算出各个image-text对的相似度之后，需要储存一些相似度比较高但不是正例的negative image-text pair 待会儿有用。

### ITM(Image-Text Matching)
将text encoder和image encoder的整个embedding编码送入multimodal encoder,经过交叉注意力和前向传播的计算之后，取其[cls]token进行二分类，这时候我们在ITC里存储的negative image-text pairs就有用了，二分类，要有正，也要有负嘛，负类就来自于那些相似度很高的neg pairs,越相似，被采样为负类的概率就越高！有老哥可能会说：ITM和ITC很像嘛，都是判断是否为一对。但这里主要的区别在于他们进行匹配的层次，ITC是在低层次的text encoder和image encoder中比较，而ITM则是在高层次的multimodal encoder中完成的哦，他们要调整的参数范围是不同滴！

### MLM(Mask Language Modeling)
经典的完型填空任务，让模型拥有一定的生成能力，但填空任务嘛，也别指望能有多强就是了。

### Momentum distillation
从网络中收集到的数据，充满噪音，正对里图片和文字的关联性可能并没有那么强，甚至是错的，动量蒸馏模型的出现是为了减少噪音的影响，可以解决我们在ITC里留的坑。动量蒸馏模型是上图左侧模型的复刻，但模型的参数是指数移动平均得出的，相当于一个更新更加缓慢、平顺的模型。在加入动量模型的损失后，这个模型的损失函数是这样滴：
![image](https://github.com/Nvnmu/Example/assets/93867177/1934e3df-f746-4e7f-9af8-c4b181af22b1)

其中 
 是原来的损失，动量模型的损失主要在右侧的KL散度，我们将图像I和文本T分别输入当前模型和动量模型，产生 
和
 是动量模型产生的image-text pairs相似度softmax后的软标签，p是当前模型的的预测概率分布，当我们在ITC里提到的问题发生时，q和p两个概率分布的KL损失会相对较小，这对模型捕捉到图像中的更多信息是有益的，因为当前模型损失计算中，只会认为对于一个image/text来说，中仅有一个标签是正确的，而分配到其他标签上的概率都会受到惩罚。但我们动量模型的损失会更注重表达的多样性。之后的MLM任务也类似。

## Conclusion
albef通过动量模型减轻了模型噪音，通过引入新的训练任务也恢复了一些模型的生成能力。

# BLIP
## Motivation
第一，之前的ALBEF模型生成能力很差，BLIP会对其进行改善。第二，针对互联网上的噪音，BLIP提出了一种新的噪音过滤和数据集生成方案。
## Architecture
![image](https://github.com/Nvnmu/Example/assets/93867177/a18358e9-62bf-41fb-b47a-81cc353dd44e)


架构上和ALBEF很像，图像encoder依然是vit,文字则是Bert,ITC和ITM训练任务是和CLIP一致的，但最后一个由MLM换为了LM 自回归任务，我们可以看到第一层的注意力由双向注意力转换为单向注意力了，这正是为了自回归caption生成做准备。
## CapFilt
![image](https://github.com/Nvnmu/Example/assets/93867177/03372790-b8c0-4e44-8b22-93eda8a6e5fa)
我们首先使用优质的人工标注数据集训练上面的ITC,ITM,LM任务，此时经过ITC和ITM训练的Encoder已经具备判断image-text pair是否为一对的能力，而Decoder则具备根据图像产生新的Caption的能力。那么面对大量的有噪音的image-text pairs时，我们可以使用Decoder进行caption形成新的image-text pairs，如果其通过了Encoder的过滤，则将这些新的pairs与高质量人工数据集进行合并，再进行预训练。

# BLIP2
之前的BLIP采取了重训模型的方法来实现多模态对齐，其采用的Bert和Vit还算比较轻量的，但如果我们要把更大参数的LLM进行多模态对齐，重新训练就太浪费资源了，而且很可能造成灾难性遗忘。所以BLIP2致力于采用一种不需要重新训练模型的方法来进行模型对齐。
## Architecture

![image](https://github.com/Nvnmu/Example/assets/93867177/45cef7ca-1524-4452-8495-533a1c986438)
BLIP2的对齐方式里，不涉及对Image Encoder 和 LLM的训练，选择了将两个大的模型进行参数对齐，训练一个Qformer完成视觉到文字的对齐工作。咱来看看这个Qformer是怎么个事儿
![image](https://github.com/Nvnmu/Example/assets/93867177/831c37c7-fa1a-4d24-85a2-cd1d2e59316e)


左侧是个Visual Transformer 负责提取特征，右侧是个Text Tranformer,即当text encoder 又当text decoder.其权重来自于Bert-base,而CA层不在Bert-base里面，所以就随机初始化了。
## Train
Stage 1第一个阶段，先不接入LLM模型，还是通过老三样ITM,ITC,ITG训练Qformer的图像到语言的转化能力。不过有些新东西，细心的同学会发现对于两个Transformer模块，其底层selfattn是共享权重的，相较于右侧的自然语言输入，左侧还有一个Learned Queries是可训练的，如果你了解Prompt Tuning的方法，会知道这是一张soft prompt，类比我们平时使用模型的时候，会为模型输入不同的指令，但就算是对同一个模型同一个任务，输入的指令稍微不一样，可能模型输出就大不相同，我们当然想找到最好效果的指令来完成图像特征提取，所以我们会用随机梯度下降来优化我们的这个Learned Queries,这个方法将离散的寻找最优自然语言提示的过程转化为了一个连续优化问题，更容易找到最好的prompt,虽然这时候我们已经不能将其转换回自然语言了，但机器看得懂就行。以下时执行一阶段任务时的掩码矩阵，匹配任务不需要掩码，对比学习时不能让Q和T相互看见，生成任务是不能偷看未来的答案。
作者：然若
链接：https://www.zhihu.com/question/495007578/answer/3441793158
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

Stage 2
第二阶段，我们接入一个LLM并冻结，将Qformer得到的视觉特征通过FFN转换维度后，利用LLM的自回归损失去训练的参数是Qformer和其后用于对齐维度的FFN。
## Dataset
利用BLIP中的CapFilt方法实现对数据的清洗。

