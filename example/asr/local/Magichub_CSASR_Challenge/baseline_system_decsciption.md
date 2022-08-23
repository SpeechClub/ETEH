# Magichub Code-Switching ASR Challenge Baseline System Description
The baseline system of Magichub Code-Switching ASR Challenge is developed using [ETEH](https://github.com/SpeechClub/ETEH). 

## Model Architecture
For the model architecture, we use Transformer. We use a 2-layer convolutional neural network (CNN) as the front-end. Each CNN layer has 320 filters, each of which has 3x3 kernel size with 2x2 stride. The self-attention encoder and decoder are 17-layer and 6-layer, respectively. All sub-layers, as well as embedding layers, produce outputs of dimension 320. In the multi-head attention networks, the head number is 8. The inner dimension of position-wise feed-forward networks is 2,048. All the ASR models are trained with batch size 512, using Adam algorithm with gradient clipping norm 5, warm-up of 25,000 steps, and Noam learning rate decay scheme. We train the ASR model for 30 epochs.

## Training Dataset and Features
We combine the MagicData-RAMC Train and TAL_CSASR Train as the training dataset for the baseline ASR model. We use 83-dimensional features, which include 80-dimensional filter banks and 3-dimensional pitch features, as input acoustic features. Features are extracted with a 25ms Hamming window, shifted every 10ms. We DO NOT apply CMVN to the acoustic features.

## Output Targets
The ASR model predicts subword units based on byte pair encoding (BPE) for English and Chinese characters for Mandarin as output targets. There are 5,276 output targets in total, of which 1,007 are English subword units (including special symbols) and 4,269 are Chinese characters. 


## Dev Set Data Pre-processing
Please refer to "dev_data_preprocess.sh" for details.

## Dev Set Scoring
Please refer to "dev_scoring_sclite.sh" for details.

## Baseline Model Performance
MER: 29.2%

Other scoring details:
<pre>
   ,-----------------------------------------------------------------------.
   |        exp/talcs_magic_160/eteh_baseline/decode/hyp.dev.gb.txt        |
   |-----------------------------------------------------------------------|
   | SPKR   | # Snt    # Chr  | Corr     Sub    Del     Ins    Err   S.Err |
   |--------+-----------------+--------------------------------------------|
   | g00    |  4456    34286  | 75.7    21.6    2.7     4.9   29.2    72.2 |
   |=======================================================================|
   | Sum/Avg|  4456    34286  | 75.7    21.6    2.7     4.9   29.2    72.2 |
   |=======================================================================|
   |  Mean  |4456.0   34286.0 | 75.7    21.6    2.7     4.9   29.2    72.2 |
   |  S.D.  |  0.0       0.0  |  0.0     0.0    0.0     0.0    0.0     0.0 |
   | Median |4456.0   34286.0 | 75.7    21.6    2.7     4.9   29.2    72.2 |
   `-----------------------------------------------------------------------'
</pre>

## [Contact](#contact)

If you have any questions, please contact us. You could open an issue on github or [email us](mailto:open@magicdatatech.com). 
