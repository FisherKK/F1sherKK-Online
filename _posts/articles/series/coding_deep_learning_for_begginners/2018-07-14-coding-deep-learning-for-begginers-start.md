---
layout: post
author: Kamil Krzyk
title:  "Coding Deep Learning for Beginners — Start!"
date:   2018-02-12 13:45:00 +0200
comments: true
categories: machine_learning deep_learning history
permalink: article/coding_deep_learning_series/:year/:month/:day/:title
---

> #### Intuition based series of articles about Neural Networks dedicated to programmers who want to understand basic math behind the code and non-programmers who want to know how to turn math into code.

This is the 1st article of series **“Coding Deep Learning for Beginners”**. You will be able to find here links to all articles, agenda, and general information about an estimated release date of next articles **on the bottom**. They are also available in [my open source portfolio — MyRoadToAI](https://github.com/FisherKK/F1sherKK-MyRoadToAI), along with some mini-projects, presentations, tutorials and links.

You can also read the article on [Medium](https://medium.com/@krzyk.kamil/coding-deep-learning-for-beginners-start-a84da8cb5044).

•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•
{: .separator}

If you read this article I assume you want to learn about one of the most promising technologies — **Deep Learning**. Statement ["AI is a new electricity"](https://medium.com/syncedreview/artificial-intelligence-is-the-new-electricity-andrew-ng-cc132ea6264) becomes more and more popular lately. Scientists believe that as *Steam Engine*, later *Electricity* and finally *Electronics* have totally changed the industry later *Artificial Intelligence* is next to transform it again. In a few years, basics of Machine Learning will become must-have skills for any developer. Even now, we can observe [increased popularity of programming languages](https://research.hackerrank.com/developer-skills/2018/) that are used mainly in Machine Learning like Python and R.

{: .header_padding_top}
## Technology that is capable of magic
In the last years applications of Deep Learning made huge advancements in many domains arousing astonishment in people that didn’t expect the technology and world to change so fast.

Let’s start from historical match between [super-computer AlphaGo](https://deepmind.com/research/alphago/) and one of the strongest Go players, 18-time world champion — Lee Sedol, in March 2016. The AI ended up victorious with the [result](https://en.wikipedia.org/wiki/AlphaGo_versus_Lee_Sedol) of 4 to 1. This match had a huge influence on Go community as AlphaGo invented completely new moves which made people try to understand, reproduce them and created totally new perspective on how to play the game. But that’s not over, in 2017 [DeepMind introduced AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/). The newer version of an already unbeatable machine **was able to learn everything without any starting data or human help. All that with computational power 4 times less** than it’s predecessor!

<img src="https://www.dropbox.com/s/q8lgeyh7fmzc9l1/i_alphago_trumph.jpeg?dl=1" width="600px">{: .image-center .image-offset-top}

{: .image-caption }
*AlphaGo versus Ke Jie in May 2017 (source: The Independent)*

Probably many of you have already heard about self-driving cars project that’s being developed for a few years, by companies like [Waymo](https://waymo.com/) (Google), [Tesla](https://www.tesla.com/autopilot), [Toyota](http://www.thedrive.com/sheetmetal/17440/toyota-to-unveil-semi-autonomous-platform-3-0-at-the-2018-consumer-electronics-show), [Volvo](https://www.volvocars.com/au/about/innovations/intellisafe/autopilot) and more. There are also [self-driving trucks](https://www.technologyreview.com/s/603493/10-breakthrough-technologies-2017-self-driving-trucks/) that are [already used on some highways in the US](https://www.wired.com/story/embark-self-driving-truck-deliveries/). Many countries slowly prepare for the introduction of autonomous cars on their roads, yet their peak is predicted for the next decade.

But how about autonomous flying car? Just recently [Udacity announced their new Nanodegree programme](https://www.udacity.com/course/flying-car-nanodegree--nd787) where developers can learn how to become Flying Car Engineers and create autonomous flying cars!

Lately thanks to improvement in AI speech recognition, voice interfaces like Google Home or Google Assistant become totally new development branch.

{% include youtube_player.html id="-qCanuYrR0g" %}{: .img .image-center}

{: .image-caption }
*Google Advertisement on Google Assistant product.*

Future when AI will inform you to leave home earlier because of traffic, buy tickets to cinema, reschedule calendar meetings, control your home and more is closer than you think.

And of course this list could be longer: [AI is capable of reproducing human speech with many dialects](https://deepmind.com/blog/wavenet-launches-google-assistant/), [AI being better at diagnosing cancer than humans](https://www.newyorker.com/magazine/2017/04/03/ai-versus-md), [AI generating new chapter of Harry Potter…](https://www.geek.com/tech/ai-generated-harry-potter-chapter-wins-the-internet-1725679/)

The key point in mentioning in all of that is making you understand that each of those inventions is using Deep Learning technology. To summarise it Deep Learning is currently excelling in tasks like:

1. Image recognition
2. Autonomous Vehicles
3. Games like Go, [Chess](https://www.theguardian.com/technology/2017/dec/07/alphazero-google-deepmind-ai-beats-champion-program-teaching-itself-to-play-four-hours), [Poker](https://www.zdnet.com/article/researchers-reveal-how-poker-playing-ai-beat-the-worlds-top-players/), but lately also [computer games](https://www.theverge.com/2017/8/11/16137388/dota-2-dendi-open-ai-elon-musk)
4. [Language Translation](https://www.theverge.com/2016/9/27/13078138/google-translate-ai-machine-learning-gnmt) (but only a few languages)
5. [Speech recognition](https://9to5google.com/2017/06/01/google-speech-recognition-humans/)
6. Analysis of handwritten texts

And this is only the beginning because technology gets democratized every day and as more people become capable of using it, the more research is being done and simple ideas tested.

{: .header_padding_top}
## So what is Deep Learning?
It’s a subset of Machine Learning algorithms, based on learning data representations, called **Neural Networks**. Basic idea is that **such  algorithm is being shown** a partial representation of reality in the form of **numerical data**. During this process, **it’s gaining experience and trying to create it’s own understanding** of given data. That understanding has **hierarchical structure** as algorithm has **layers**. First layer learns the simplest facts and is connected to the next layer that uses experiences from previous one to learn more complicated facts. **Number of layers is called the depth of the model.** The more layers, the more complicated data representations the model can learn.

<img src="https://www.dropbox.com/s/1vanyrd6obt5v46/i_nn_example.png?dl=1" width="800px">{: .image-center .image-offset-top}

{: .image-caption }
*Neural Network that is used for face detection. It learns hierarchy of representations: corners in first layer, eyes and ears in the second layer, and faces in the third layer (source: strong.io)*

## Is Deep Learning really new technology?
Some of you might think that Deep Learning is technology that was developed lately. That’s not entirely true. Deep Learning had very rich history and had various names depending on philosophical viewpoint. People were [dreaming about intelligent machines over a hundred years ago before first mathematical concepts were built](https://en.wikipedia.org/wiki/Ada_Lovelace). There have been three waves of development.

During the first wave, Deep Learning went by name [Cybernetics](https://en.wikipedia.org/wiki/Cybernetics). First predecessors of modern deep learning were linear models inspired by the study about the nervous system—[Neuroscience](https://en.wikipedia.org/wiki/Neuroscience). The first [concept of the neuron (1943)](https://pdfs.semanticscholar.org/5272/8a99829792c3272043842455f3a110e841b1.pdf), the smallest piece of Neural Network, was proposed by McCulloch-Pitt that tried to implement brain function. A few years later Frank Rosenblatt turned that concept into the first trainable model—[Mark 1 Perceptron](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.335.3398&rep=rep1&type=pdf).

<img src="https://www.dropbox.com/s/rxivf3o8fvihtja/i_mark_perceptron.jpeg?dl=1">{: .image-center .image-offset-top}

{: .image-caption }
*Mark 1 Perceptron (source: Wikipedia)*

But people had problems to describe brain behaviours with theories available at that time. That’s why interest in them decreased for the next 20 years.

The second wave started in the 80s and went by name [Connectionism](https://en.wikipedia.org/wiki/Connectionism) but also term Neural Networks started to be used more often. The main idea was that neurons could achieve more intelligent behaviours when grouped together in large number. This concept was introduced by Hinton and is called [distributed representation (1986)](https://www.cs.toronto.edu/~hinton/absps/families.pdf). It’s still very central to today’s Deep Learning. Another great accomplishment of a second wave was the invention of [back-propagation by Yann LeCun (1987)](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf)—core algorithm that is used until today for training Neural Network parameters. Also in year 1982 [John Hopfield](https://en.wikipedia.org/wiki/John_Hopfield) has invented Recurrent Neural Networks, which after additional introduction of [LSTM in 1997, are used today for language translation](http://www.bioinf.jku.at/publications/older/2604.pdf). Those few years of big hype about Neural Networks has ended due large interest of the various investors which expectations towards implementing AI in products was not fulfilled.

<img src="https://www.dropbox.com/s/am9rp0bduf7gcjh/i_lstm.png?dl=1" width="800px">{: .image-center .image-offset-top}

{: .image-caption }
*Image of LSTM cell based Recurrent neural Network (source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/)*

The third wave started in 2006. At that time computer became a more common thing that everyone could afford. Thanks to the various groups, e.g. gamers, has grown the market for powerful GPUs. Internet was available to everyone. Companies started paying more attention to analytics — gathering data in digital form. As a side effect researchers had more data, and computational power to perform experiments and validate theories. Consequently, there was another huge advancement, thanks to [Geoffrey E. Hinton that managed to train Neural Network with many layers](http://www.cs.toronto.edu/~fritz/absps/ncfast.pdf). From that moment a lot of different proposals for Neural Network architectures with many layers started to appear. Scientists referred to the number of layers in Neural Network as of “depth” — the more layers it had the deeper it was. Very important occurring was usage of Convolutional Neural Network [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) in image classification contest ILSVRC-2012. It has revolutionized a lot of industries by providing them with reliable image detection mechanism — allowing many machines to see e.g. autonomous cars.

<img src="https://www.dropbox.com/s/1pxt9pmlx0imhbu/i_alexnet.jpeg?dl=1" width="800px">{: .image-center .image-offset-top}

{: .image-caption }
*Structure of AlexNet CNN (source: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks”, 2012)*

In 2014, Ian Goodfellow has introduced new type of Neural Networks called [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). In this architecture two Neural Networks were competing against each other. First network tried to mimic some distribution of data. The role of a second network was to tell if the data it received is fake or real. The goal of first network is to trick the second network. Competition lead to increase in performance of first network and made generation of any kind of data — images, music, text, speech possible.

<img src="https://www.dropbox.com/s/giod5qruw7p256z/i_fox_gan.gif?dl=1" width="800px">{: .image-center .image-offset-top}

{: .image-caption }
*GAN used to transfer style of one image into another (source: https://github.com/lengstrom/fast-style-transfer)*

And this is it I guess. Third wave continues until today and it depends on us how far it can go!

{: .header_padding_top}
## Why am I creating this series of articles?
I am really passionate about Machine Learning and especially Deep Learning. My dream is to become Machine Learning Expert — person who work with people to solve problems and democratize the knowledge. I am working hard every day to reach that goal and this blog is a part of it. So study with me!

In my opinion, the biggest problem with access to this technology is that it was developed at universities and in laboratories by highly qualified Ph.D scientists and still partially stays there. It’s understandable as everything is strongly based on [Linear Algebra](https://en.wikipedia.org/wiki/Linear_algebra), [Probability](https://en.wikipedia.org/wiki/Probability) and [Information Theory](https://en.wikipedia.org/wiki/Information_theory), [Numerical Computing](https://en.wikipedia.org/wiki/Numerical_analysis). But **in order to become a driver you don’t need to know the engine** right? There is still conviction that in order to work in this field you need to be Ph.D but it is [starting to change in terms of Software Engineering](https://www.quora.com/Is-a-PhD-necessary-for-a-job-in-machine-learning-or-can-I-work-in-the-industry-without-one-Would-I-still-be-able-to-work-on-the-cutting-edge-Is-a-PhD-worth-it-if-I-have-no-intentions-of-entering-academia).

Demand for people with those skills will become so big it will simply become impossible for everyone to have Ph.D title. That’s why in order to make people use it, there must be someone who can translate that to others, while skipping complicated proofs, scientific notation and adding more intuition.

{: .header_padding_top}
## What I hope to show to you
My goal is to provide strong understanding of most popular topics related to Deep Learning. I don’t want to be protective when it comes to picking content — I want to show you even more complicated stuff and at the same time, do my best to provide you with intuition to grasp it. My main priority is to allow you understand to how those algorithms work and teach you how to code them from scratch. Like Mark Daoust (Developer Programs Engineer for TensorFlow) once said to me:

> Everyone should code Neural Network from scratch once… but only once…

So there will be a lot of code that I plan to carefully explain. Among the topics, you can expect mini-projects where I will show you how to use what we’ve learned. **It’s really important for knowledge to be followed by practice.**

The approach will be bottom-up then:

- low-level — basic (and explained) math turned into Python NumPy code,
- mid-level — TensorFlow (both tf.nn and tf.layer modules) where most of the stuff that I’ve already shown to you can be automated in a single line of code,
- high-level — very popular framework that allows you to create Neural Networks really fast — Keras.

This project will focus only on Multilayer Perceptrons. It’s already a lot of work to be done. If it succeeds I might consider doing an extension for Convolutional Neural Networks, Recurrent Neural Networks, and Generative Adversarial Neural Networks.

## Next Article
The next article is available [here](https://kamilkrzyk.com/article/coding_deep_learning_series/2018/07/25/coding-deep-learning-for-begginers-types-of-machine-learning).
