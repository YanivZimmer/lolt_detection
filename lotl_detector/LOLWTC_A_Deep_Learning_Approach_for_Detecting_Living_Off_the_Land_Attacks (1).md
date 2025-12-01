## Page 1

Proceedings of CCIS2023

# LOLWTC: A deep learning approach for detecting Living off the Land attacks

**Kuiye Ding¹, Shuhui Zhang²*, Feifei Yu², Guangqi Liu²**

¹Qilu University of Technology(Shandong Academy of Science), Jinan 250353, China
²Qilu University of Technology (Shandong Academy of Sciences), Shandong Computer Science Center (National Supercomputer Center in Jinan), Shandong Provincial Key Laboratory of Computer Networks, Jinan 250014, China
202185010006@stu.qlu.edu.cn; *zhangshh@sdas.org; yuff@sdas.org; liuguangqi@sdas.org

**Abstract:** Living off the Land (LotL) attacks have gained attention in recent years because they are sneaky. These attacks exploit legitimate tools, scripts, and system permissions, making them hard to detect and track. As a result, defense costs increase. However, most research focuses on detecting and classifying malware rather than LotL attacks. This study aims to explore a novel approach that combines machine learning, deep learning, and natural language processing methods for detecting LotL attacks. We propose a deep learning detection framework called LOLWTC. It utilizes word embedding to represent qualitative features in command-line text as two-dimensional matrices. These matrices are then used for classification using deep networks. Experimental results demonstrate the robustness and effectiveness of LOLWTC. It achieves an impressive f1 score of 0.9945 on the test set during 10-fold cross-validation. This showcases its significant potential for detecting LotL attacks and addressing associated security concerns.

**Keywords:** LOLWTC, Living off the Land (LotL), Commands, Command-Line, Machine Learning, Deep Learning

## 1 Introduction

LotL attacks utilize legitimate tools and system components to control targeted systems without being detected[4]. However, detecting these attacks is challenging because there is no malware involved and minimal traces are left behind. Defending against them is costly, requiring advanced detection methods and strong security protocols. To effectively protect organizations from LotL attacks, a comprehensive approach is necessary.

This paper focuses on using deep learning to detect LotL attacks. In the field of malware detection, Li et al. (2020) propose an adversarial machine learning method that emphasizes the use of OpCode n-grams feature to enhance the performance of malware detection models[1]. Penmatsa et al. (2020) contribute to finding efficient detection features by identifying the minimum feature set for malware detection. They highlight the importance of feature selection for effective and efficient detection systems [2]. Gibert et al. (2020) conduct a comprehensive examination of methods and features used in traditional machine learning workflows for malware detection and classification. They particularly focus on recent trends and developments in deep learning methodologies [3]. Barr-Smith et al. (2021) perform an extensive systematic investigation, the first of its kind, on the utilization of these techniques by malware specifically targeting Windows systems[4]. Bellizzi et al. (2021) introduce a comprehensive conceptual framework exploring the intricacies and implications of just-in-time (JIT) multi-factor (MF) drivers[5]. Ongun et al. (2021) propose LOLAL, a meticulously designed Active Learning framework for detecting LOL attacks[6]. Shukla (2022) explores the utilization of Resistive Random Access Memory (RRAM) as a defense mechanism against security threats. These works collectively contribute to the advancement of security algorithms, malware detection, and understanding the functionalities and dynamics associated with malware attacks [7].

This study proposes a LotL attack detection method that utilizes Word2vec and TextCNN. Firstly, historical command-line data is collected from Linux systems. The collected data is then trained using Word2vec to construct a word vector library. Subsequently, the TextCNN model is employed to classify the command-line data. During the training process, the neural network's weight parameters are optimized based on the data's features to achieve accurate LotL attack detection. This method utilizes deep learning techniques to improve the accuracy and robustness of the classification.

Additionally, this study incorporates manual detection approaches for LotL attacks and presents a LotL detection method that utilizes command-line labels and machine learning. It also provides a comparison with detection methods based on deep learning.

## 2 The method based on Command-line Labels and Machine Learning

In their 2022 study[8], Boros et al. examined the use of machine learning to detect LotL attacks. They discussed the challenges associated with distinguishing between legitimate and malicious activities, as well as the limited representation of LotL detection in anti-virus software. Building on this research, our study further explores the influence of different natural language processing methods on detecting LotL attacks.

---

979-8-3503-0442-8/23/$31.00 ©2023 IEEE

Authorized licensed use limited to: Bar Ilan University. Downloaded on November 30,2025 at 19:14:02 UTC from IEEE Xplore. Restrictions apply.
&lt;page_number&gt;176&lt;/page_number&gt;

---


## Page 2

Proceedings of CCIS2023

&lt;img&gt;Diagram showing command line feature construction.&lt;/img&gt;
Mark specific tags according to the characteristics of the command line.

&lt;img&gt;Diagram showing tagging context.&lt;/img&gt;
Tagging Context

Command Line

Figure 1 Feature Construction Diagram

**2.1 Command line feature construction**

Feature extraction is performed through manual analysis to determine if command lines are used for LotL attacks. This analysis involves identifying different characteristics, including the binary files used, the parameters employed, the environment variables set, and whether network access is involved.

**Network:** Network features are sought in all command lines, such as network protocols (e.g., HTTP, FTP) or IP addresses. For example: "curl http://example.com/api" or "traceroute 192.168.1.1".

**Binary:** If a command line includes system files that can be exploited for LotL attacks, they are marked accordingly. For instance: "env /bin/sh" or "expect -c 'spawn /bin/sh; interact'".

**KEYWORD_-c:** The "-c" parameter plays a critical role in many system files, and its usage in a command line is recorded. For example: "gem open -e '/bin/sh -c /bin/sh' rdoc" or "tar xvzf /home/Downloads/ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin".

**COMMAND_INSTALL:** If an "install" command is present, it indicates the command is likely involved in downloading files, and it is labeled accordingly. For instance: "yarn install --force" or "sudo install -m =xs $(which ab) .".

A total of fourteen features are generated from the aforementioned characteristics. The process of constructing these features is illustrated in Figure 1, while Figure 2 presents an example of the annotation.

**2.2 Text Vector Transformation**

Text cannot be directly used for model training, so it needs to be transformed into vector form. This study utilizes five methods to transform text into vectors:

**Bag-of-Words (BoW):** is a text representation method that considers the frequency of word occurrences. It treats text as a collection of words without considering their order. BoW represents text as a frequency vector, where each dimension corresponds to a word and the value represents the word's frequency in the text.

**TF-IDF (Term Frequency-Inverse Document Frequency):** is a widely used method for representing text. It assesses the significance of a word in a collection of documents. TF-IDF combines term frequency (word occurrences in a document) and inverse document frequency (logarithm of the number of documents containing the word) to calculate a value indicating the word's importance in a document.

**Word2Vec** is a neural network-based algorithm that converts words into low-dimensional vector representations. It trains a two-layer neural network to map words to vectors in a vector space, allowing words with similar meanings to have closer vector representations. Word2Vec can be implemented using Continuous Bag-of-Words (CBOW) or Skip-Gram approaches.

&lt;img&gt;Diagram showing text vector transformation methods.&lt;/img&gt;
Natural Language Processing Method

- Bag-of-Words
- TF-IDF
- Word2vec
- Doc2vec
- WV-TI

**Doc2Vec** is a neural network-based algorithm that converts entire documents into vector representations. It trains a two-layer neural network to map documents to vectors in a vector space. Doc2Vec can be implemented using PV-DM (Paragraph Vector - Distributed Memory) or PV-DBOW (Paragraph Vector - Distributed Bag of Words) approaches.

&lt;page_number&gt;177&lt;/page_number&gt;
Authorized licensed use limited to: Bar Ilan University. Downloaded on November 30,2025 at 19:14:02 UTC from IEEE Xplore. Restrictions apply.

---


## Page 3

Proceedings of CCIS2023

**WV-TI** is a method that combines Word2Vec and TF-IDF. It converts words into vectors using Word2Vec and calculates the TF-IDF weight for each word in each document. The word vectors are then multiplied by their corresponding TF-IDF weights, and the weighted average of all word vectors is computed to obtain the document vector representation. This method improves text representation quality by incorporating semantic information from Word2Vec and weight information from TF-IDF.

## 2.3 Application of Machine Learning Models

By employing Random Forest as the detection model, we analyze the performance of the model using different text vector transformation methods. The results obtained from ten-fold cross-validation are as follows.

**Validation classification metrics:** When detecting malicious command lines, we use four evaluation metrics for classification: accuracy, precision, recall, and F-measure. The F-measure is a metric that simultaneously reflects both accuracy and recall.

$$Recall = \frac{TP}{TN + FP}$$ (1)
$$Precision = \frac{TP}{TN}$$ (2)
$$F - measure = 2 * \frac{Precision*Recall}{Precision+Recall}$$ (3)
$$Accuracy = \frac{TP + TN}{TP + FP + TN + FN}$$ (4)

Where TP represents the number of correctly predicted malicious command-lines, FP represents the number of benign command-lines incorrectly predicted as malicious, TN represents the number of correctly predicted benign command-lines, and FN represents the number of truly malicious command-lines incorrectly predicted as benign.

**Table I** The performance of the Random Forest model was evaluated using ten-fold cross-validation under various natural language processing methods.

<table>
<thead>
<tr>
<th>Method</th>
<th>Accuracy</th>
<th colspan="2">Macro Precision</th>
<th colspan="2">Macro Recall</th>
<th colspan="2">Macro F1-score</th>
</tr>
</thead>
<tbody>
<tr>
<td>BOW</td>
<td>0.9917</td>
<td>0.9619</td>
<td>0.9670</td>
<td>0.9638</td>
<td></td>
<td></td>
</tr>
<tr>
<td>TF-IDF</td>
<td>0.9927</td>
<td>0.9805</td>
<td>0.9557</td>
<td>0.9673</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Word2vec</td>
<td>0.9812</td>
<td>0.9424</td>
<td>0.8913</td>
<td>0.9140</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Doc2vec</td>
<td>0.9787</td>
<td>0.9660</td>
<td>0.8406</td>
<td>0.8975</td>
<td></td>
<td></td>
</tr>
<tr>
<td>WV-TI</td>
<td>0.9834</td>
<td>0.8953</td>
<td>0.9688</td>
<td>0.9305</td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

When considering both overall accuracy and the time required for model training, TF-IDF stands out as an excellent option. It achieves high model accuracy while keeping the training time relatively low. However, it does have limitations when it comes to the F1 score, suggesting the need for further improvement.

## 3 The method based on Deep Learning

Next, this paper applies two approaches for LotL attack detection. One is based on the BERT pre-trained model, and the other is based on LOLWTC.

### 3.1 BERT pre-trained model with fully connected neural network

BERT, also known as Bidirectional Encoder Representations from Transformers, is an incredibly successful pre-trained language model developed by Google[9]. It has achieved state-of-the-art performance in various natural language processing tasks. BERT leverages the Transformer architecture to learn bidirectional contextual word representations. By utilizing large-scale unlabeled text data for self-supervised pretraining, BERT acquires comprehensive representations capturing syntax and semantics. Its effectiveness is further enhanced by fine-tuning on labeled data specific to the tasks at hand.

To apply BERT for LotL attack detection, we load the pre-trained BERT model and tokenizer. The tokenizer converts input command lines into vectors, and the BERT model obtains encoded representations. The attention mechanism in BERT plays a crucial role in capturing contextual information, which is calculated using the attention formula:

$$Attention(Q, K, V) = softmax(\frac{Q \times K^T}{sqrt(d_k)}) \times V$$ (5)

Next, we construct a multi-input neural network with an

&lt;img&gt;Figure 2 ROC curve&lt;/img&gt;
Figure 2 ROC curve

&lt;img&gt;Figure 4 The time required for model training under various methods.&lt;/img&gt;
Figure 4 The time required for model training under various methods.

&lt;page_number&gt;178&lt;/page_number&gt;
Authorized licensed use limited to: Bar Ilan University. Downloaded on November 30,2025 at 19:14:02 UTC from IEEE Xplore. Restrictions apply.

---


## Page 4

Proceedings of CCIS2023

input feature dimension of 768. The network comprises two hidden layers with sizes of 128 and 64, respectively. The output layer consists of two classes. We use ReLU as the activation function and cross-entropy as the loss function. Stochastic gradient descent (SGD) is applied as the optimizer with a learning rate of 0.01. The network undergoes 100 training iterations.

$$ReLU(x) = max(0,x) \quad (6)$$

### 3.2 LOLWTC: LotL Attack Detection Method Based on Word2Vec and TextCNN

The TextCNN model comprises convolutional layers, pooling layers, and fully connected layers[10]. The convolutional layers aim to extract textual features, while the pooling layers reduce the dimensionality of the features. The fully connected layers are responsible for the classification prediction.

When defining the model, several choices are made, such as the size of convolutional kernels, pooling methods, activation functions, and other parameters. The model is optimized using the Adam optimization algorithm. The training data is fed into the TextCNN model, and the network parameters are updated through forward and backward propagation calculations until the loss function converges or a predetermined number of iterations is reached.

In this study, we use the word2vec approach to generate a matrix for each command-line text. The matrix size is set to 100×400, where 400 representing the dimensionality of the word vectors. Although all word vectors have the same dimensionality, the number of words in each text may vary. If the number of words exceeds 100, truncation is done, and if the number is less than 100, padding with zeros is applied.

The convolutional layers in the TextCNN model utilize five convolutional kernels: (2x1), (3x1), (4x1), (5x1), and (6x1). Following the convolutional operations, the Rectified Linear Unit (ReLU) activation function is applied.

matrix in each region and concatenate them to form a vector. This vector is subsequently utilized for classification prediction within the fully connected neural network.

**Table II** The model performance of BERT and LOLWTC under ten-fold cross-validation.

<table>
<thead>
<tr>
<th>Method</th>
<th>Accuracy</th>
<th colspan="2">Macro Precision</th>
<th colspan="2">Macro Recall</th>
<th>Macro F1-score</th>
</tr>
</thead>
<tbody>
<tr>
<td>BERT</td>
<td>0.9943</td>
<td>0.9948</td>
<td></td>
<td>0.9778</td>
<td></td>
<td>0.9859</td>
</tr>
<tr>
<td>LOLWTC</td>
<td>0.9945</td>
<td>0.9947</td>
<td></td>
<td>0.9945</td>
<td></td>
<td>0.9945</td>
</tr>
</tbody>
</table>

Based on the results, both BERT and LOLWTC have shown promising performance. However, LOLWTC has a slightly higher F1 measure compared to the BERT model, with an increase of approximately 1 percentage point. Additionally, LOLWTC demonstrates a higher recall rate, showing an increase of around 2 percentage points. Therefore, the LOLWTC model exhibits superior performance.

## 4 Experiments

&lt;img&gt;Figure 4 TextCNN Model Diagram&lt;/img&gt;

**Figure 4** TextCNN Model Diagram

&lt;img&gt;Figure 3 Model Specific Parameters&lt;/img&gt;

**Figure 3** Model Specific Parameters

The model is designed with four hidden layers. Each region generates four matrices, which are then transformed into matrices of size (matrix dimension x 1) using convolutional kernels of size (1 x word vector dimension).

The pooling layers extract the maximum values from each

### 4.1 Operating Environment

The hardware used in our experiment consists of an Intel(R) Core(TM) i5-11400H processor with 16GB of memory and an NVIDIA GeForce RTX 3050Ti graphics card. The software environment includes a 64-bit Windows 10 operating system and VMWare, which

&lt;page_number&gt;179&lt;/page_number&gt;
Authorized licensed use limited to: Bar Ilan University. Downloaded on November 30,2025 at 19:14:02 UTC from IEEE Xplore. Restrictions apply.

---


## Page 5

Proceedings of CCIS2023

installs the Ubuntu 20.04 virtual machine for testing malicious samples. The environment for building and running the deep learning framework comprises Python 3.8, Miniconda 22.11.1, and PyTorch torch.12.1.

## 4.2 Dataset construction

Our task is to develop a model that can analyze raw command-line inputs and identify whether they are being used for LotL attacks. To begin with, we need to construct a suitable dataset for model training. Currently, there is no publicly available dataset specifically designed for this purpose. However, obtaining a malicious dataset from online sources is relatively straightforward. The LOLBAS website (https://lolbas-project.github.io/) provides a comprehensive collection of files and command lines used for exploiting LotL attacks on Linux systems. Since there are differences between Windows and Linux systems, our experiment focuses exclusively on detecting command lines in Linux. In the end, we collected a total of 1,242 instances of malicious command lines.

To find benign data, we initially focused on detecting LotL attacks on servers belonging to individual developers and small to medium-sized enterprises. Based on this, we collected commonly used Linux commands and historical command lines from servers used by individual developers. To make the dataset more representative of real-world scenarios, we aimed to obtain an imbalanced dataset. As a result, we gathered 18,278 command lines as benign data, which were then filtered, leaving us with 9,896 remaining instances.

## 4.3 Comparison of Different Models

As benign command-line samples are sensitive and there are currently no publicly available datasets, we opted to use a publicly available dataset of malicious command-line samples for this study. Although there are some differences in the datasets used between our research and others due to confidentiality reasons, LOLWTC has demonstrated practical value in terms of its effectiveness. The evaluation metric used in the two existing research articles[6][8] is F1-score. Due to limited research in this area, our study only compares two previous research investigations.

Table III Comparison of Different Models

<table>
<thead>
<tr>
<th>Method</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>Ongun (2021)[6]</td>
<td>0.9100</td>
</tr>
<tr>
<td>Boros T(2022)[8]</td>
<td>0.9500</td>
</tr>
<tr>
<td>Our Model</td>
<td>0.9945</td>
</tr>
</tbody>
</table>

In addition, we conducted an analysis of misclassified samples and observed that some malicious samples have little distinction from benign samples. From this perspective, relying solely on command-line text as the sole basis for detecting off-ground attacks is relatively limited. Future research should explore combined approaches using multiple detection methods. These methods can integrate command-line features, system behavior monitoring, network traffic analysis, and other data sources to enhance the comprehensiveness and accuracy of detection models. Such comprehensive approaches will better capture the characteristics of off-ground attacks and reduce the possibility of false positives. Further research could consider incorporating deep learning techniques to automatically learn and uncover more complex off-ground attack patterns, continuously improving the performance of detection systems. Within feasible limits, real-time monitoring and alert mechanisms can be developed to enhance the system's response capability against potential off-ground attacks. These efforts will provide a more comprehensive and reliable protection for the security of the Linux system.

## 5 Conclusions

This study highlights the growing concern among researchers and practitioners regarding the stealthy nature of "Living off the Land" (LotL) attacks. These attacks present challenges in terms of detection and tracking, leading to increased defense costs. While previous research has mainly focused on malware detection and classification, the detection of LotL attacks has received limited attention. To address this research gap, we propose a deep learning detection framework called LOLWTC, which utilizes natural language processing techniques for LotL attack detection.

Our proposed method employs word embedding and TextCNN to represent qualitative features in command-line text as two-dimensional matrices. These matrices are then used for classification through deep neural networks. Experimental results demonstrate the remarkable potential of LOLWTC, achieving an impressive F1 score of 0.9945 on the test set under 10-fold cross-validation. This highlights its effectiveness in detecting LotL attacks and mitigating security concerns.

Furthermore, we explore the impact of different natural language processing methods on LotL attack detection. We also compare the performance of deep learning-based detection methods with manual detection approaches. The study concludes that deep learning surpasses traditional machine learning in LotL attack detection. Among the deep learning methods, LOLWTC outperforms the BERT model, exhibiting slightly higher F1 measure and recall rate. This underscores its practical potential for LotL attack detection.

## Acknowledgements

The author of this article is very grateful for the guidance of Teacher Zhang, which deeply influenced the technical solutions and writing process of this article. This work was supported by the National Natural Science Foundation of China (62102209).

## References

[1] Xiang Li; Kefan Qiu; Cheng Qian; Gang Zhao; "An Adversarial Machine Learning Method Based on OpCode N-grams Feature in Malware Detection", 2020 IEEE FIFTH INTERNATIONAL CONFERENCE ON DATA SCIENCE IN ..., 2020. (IF: 3)

&lt;page_number&gt;180&lt;/page_number&gt;
Authorized licensed use limited to: Bar Ilan University. Downloaded on November 30,2025 at 19:14:02 UTC from IEEE Xplore. Restrictions apply.

---


## Page 6

Proceedings of CCIS2023

[2] Ravi Kiran Varma Penmatsa; Akhila Kalidindi; S. Kumar Reddy Mallidi; "Feature Reduction and Optimization of Malware Detection System Using Ant Colony Optimization and Rough Sets", INT. J. INF. SECUR. PRIV., 2020.

[3] Daniel Gibert; Carles Mateu; Jordi Planes; "The Rise of Machine Learning for Detection and Classification of Malware: Research Developments, Trends and Challenges", J. NETW. COMPUT. APPL., 2020. (IF: 5)

[4] Frederick Barr-Smith; Xabier Ugarte-Pedrero; Mariano Graziano; Riccardo Spolaor; Ivan Martinovic; "Survivalism: Systematic Analysis of Windows Malware Living-Off-The-Land", 2021 IEEE SYMPOSIUM ON SECURITY AND PRIVACY (SP), 2021. (IF: 3)

[5] Jennifer Bellizzi; Mark Vella; Christian Colombo; Julio Hernandez-Castro; "Responding to Living-Off-the-Land Tactics Using Just-in-Time Memory Forensics (JIT-MF) for Android", ARXIV-CS.CR, 2021.

[6] Talha Ongun; Jack W. Stokes; Jonathan Bar Or; Ke Tian; Farid Tajaddodianfar; Joshua Neil; Christian Seifert; Alina Oprea; John C. Platt; "Living-Off-The-Land Command Detection Using Active Learning", ARXIV-CS.CR, 2021.

[7] Sanket Shukla; "Design of Secure and Robust Cognitive System for Malware Detection", ARXIV-CS.CR, 2022.

[8] Boros T, Cotaie A, Stan A, et al. Machine Learning and Feature Engineering for Detecting Living off the Land Attacks[C]/IoTBDS. 2022: 133-140.

[9] Devlin J , Chang M W , Lee K , et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[J]. 2018.

[10] Kim Y . Convolutional Neural Networks for Sentence Classification[J]. Eprint Arxiv, 2014

&lt;page_number&gt;181&lt;/page_number&gt;
Authorized licensed use limited to: Bar Ilan University. Downloaded on November 30,2025 at 19:14:02 UTC from IEEE Xplore. Restrictions apply.