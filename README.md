# Large Behavior Model on Twin-2K-500
Designing a Large Behavior Model for the following problem statement


**Given a person's answers from waves 1–3 (their "persona"), predict how that same person answers the held-out wave-4 questions — and evaluate against both the real wave-4 answers and the human test–retest ceiling**


# Section 1 Data Exploration 

Dataset Link - https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500

#### Dataset Usage
1. Full Persona Dataset - This dataset contains all waves data reported together. It can be skipped as the persona information columns contains wave 4 responses for common set of questions of wave 1-3 and wave 4. This would lead to target leakage if used for training.
2. **Wave Split Dataset** - After analyzing the dataset and also as reported by the authors, this is the dataset I used for building and evaluating models.
3. Question Catalog (question_catalog.json) - This datasets contains all the Question asked in either wave 1-3 or wave 4 to the participant. I used this dataset for evaluation metrics purposes to create answer ranges from the answer options for each question based on their `QuestionID` field. Specifically for Matrix and MC Type Questions available via `QuestionType` field, I used `Col` and `Options` fields respectively to get the available set of answers for each question.
  
### Dataset Statistics - Wave Split Dataset

###### Token statistics
- Why ? To help decide which model to choose in terms of context length.
- Full Persona - For summary text of a persona profile, maximum of 3,061 words and average of 2,014 words, can be approximated as X tokens based on 1 token = 3/4 word 
- Wave Dataset - Answer are around similar length for wave1-3 and wave 4 and are around 6320 words.



### Analysis of Q&A of Wave 1 to Wave 3

I analayzed the Wave 1 to Wave 3 Q&A from the `wave1_3_persona_json` by expanding the json format column from the `wave_splits `dataset which is at persona or user level to each Q&A answer by each user. 
The statistics are as follows  - 

- Total Questions Answered :  353929 
- Unique Number of Persons : 2058
- Total number of Duplcates Q&A rows Found - 0
- Total Uniques Questions Asked - 171. The following table shows a breakdown of Question by different block names - 

| Block Name                      | Unique Question Counts | Question Types          | 
|---------------------------------|-----------------------:|-------------------------|
| Cognitive tests                 | 69                     | DB, MC, TE              |
| Personality                     | 49                     | DB, MC, Matrix, TE      |
| Economic preferences            | 36                     | DB, MC, Matrix, TE      |
| Demographics                    | 14                     | MC                      |
| Economic preferences - intro    | 2                      | DB                      |
| Forward Flow                    | 1                      | TE                      |
| Total | 171 | | 

Where DB = Descriptive Bloc, MC = Multiple Choice, TE = Text Entry and Matrix = Likert scale

- The dataset is heavily skewed towards cognitive test and lesser towards the economic preferences. Keeping the representations of each question is quite important if we were to sample questions to fit the context size of an LLM model as any question can be asked at test time. 

- The following table shows the dataset contribution by different Question Types 

| Question Type | Number of Rows | Percentage (%) |
|---------------|---------------:|---------------:|
| MC | 220,199 | 62.22 |
| Matrix | 59,682 | 16.86 |
| TE | 45,236 | 12.78 |
| DB | 28,812 | 8.14 |

- The dataset is heavily skewed towards Multiple Choice Quesstion and lesser towards the Matrix type question. Keeping the representations of each question type is quite important if we were to sample questions to fit the context size of an LLM model as any question can be asked at test time. 

### Analysis of Q&A of Wave 4 

For wave 4 Q&A extraction, I used the `wave4_Q_wave4_A` field of the wave_split dataset and unroll it to prepare data at Q&A-PID level
- Total Number of Rows which represent number of Q&A answered in total - 131712
- Total Uniques Questions Asked - 85. The following table shows a breakdown of Question by different block names - 

| Block Name | Unique Question Count | Question Types |
|------------|----------------------:|----------------|
| Product Preferences - Pricing | 41 | DB, MC |
| Non-experimental heuristics and biases | 5 | MC, Matrix, Slider |
| Anchoring - African countries high | 2 | MC, TE |
| Anchoring - African countries low | 2 | MC, TE |
| Anchoring - redwood high | 2 | MC, TE |
| Anchoring - redwood low | 2 | MC, TE |
| Proportion dominance 1C | 1 | MC |
| Outcome bias - success | 1 | MC |
| Probability matching vs. maximizing - Problem 1 | 1 | Matrix |
| Probability matching vs. maximizing - Problem 2 | 1 | Matrix |
| Proportion dominance 1A | 1 | MC |
| Proportion dominance 1B | 1 | MC |
| Absolute vs. relative - calculator | 1 | MC |
| Proportion dominance 2A | 1 | MC |
| Proportion dominance 2B | 1 | MC |
| Outcome bias - failure | 1 | MC |
| Sunk cost - no | 1 | TE |
| Sunk cost - yes | 1 | TE |
| WTA/WTP Thaler - WTP noncertainty | 1 | MC |
| WTA/WTP Thaler problem - WTA certainty | 1 | MC |
| Proportion dominance 2C | 1 | MC |
| Myside Ford | 1 | MC |
| Myside German | 1 | MC |
| Absolute vs. relative - jacket | 1 | MC |
| Linda-conjunction | 1 | Matrix |
| Linda - no conjunction | 1 | Matrix |
| Less is More Gamble C | 1 | MC |
| Less is More Gamble B | 1 | MC |
| Less is More Gamble A | 1 | MC |
| False consensus | 1 | Matrix |
| Disease-loss | 1 | MC |
| Disease - gain | 1 | MC |
| Base-rate 70 engineers | 1 | Slider |
| Base-rate 30 engineers | 1 | Slider |
| Allais Form 2 | 1 | MC |
| Allais Form 1 | 1 | MC |
| WTA/WTP Thaler problem - WTP certainty | 1 | MC |



#### Other Findings
- DB questions are descriptive blocks for the next question and should not be considered for evaluation.
I removed 2058 rows from the test-retest accuracy evaluation set. But these block could be useful for training data as they provide LLMs with the context for a question
- Users can also skip the question 
    - We can anayze user Settings -> ForceResponse 
    - And remove the skipped question (Didn't completed)


### Test-Retest Evaluation of reliability of humans

#### Step 1. Filtering data for evaluation 
I filtered the wave 4 dataset which contains 131,712 Q&A pairs rows.
- Filter 1 After Dropping DB descriptive block question's 2058 rows, remaining rows were - 129654
- Filter 2 I also dropped questions where both Wave 1 to Wave3 and Wave 4 answer are None.
There were around 10,290 rows found which belonged to the TE (6,174) rows and Slider Question types (4,116) rows

After apply filter 1 and filter 2, we are left with 119,364 Q&A rows answered by 500 participants in Wave 4 for human test-retest evaluation dataset.

The statistics of the dataset by quesion type is as following
|  Question Type   | Number of Questions 
|-----|-----| 
|Multiple Choice (MC) | 109,074 | 
|Matrix | 10,290
|Total | 119,364

#### Step 2. Evaluation Methodology

As Evaluation will differ by Question Types because they have different answer ranges, we must compute metrics for them individually. As reported in the paper the answer are of 2 types binary or numerical. 

Author Evaluation Metric Definition 
 ```
 For non-binary measures, we calculate the absolute deviation between the ground truth and predicted answer, divided by the range of possible answers. We then compute accuracy as 1 minus this absolute deviation. This measure generalizes accuracy from binary to numerical questions: it ranges between 0 and 1, is equal to 1 when the prediction is equal to the ground truth, and 0 when it is maximally different. When multiple questions are included in the same task, we take the mean accuracy across the questions within each task
 ```

As we have 2 types Multiple Choice (MC) and Matrix Type, I handled them in following ways
- MC(Multiple Choice) Question type - It can be binary or range of answers. 
    - For binary questions, I consider the answer as correct and score of 1 if the Wave4 answer exact match of Wave1-3 Answer else its a score of 0.
    - For MC Question 

- For Matrix Question types, which we have 36 unique questions. Out of them, 23 questions are likert scale and 13 bipolar.For these questions, I subset their QuestionId from the `question_catalog.json`. I then used their `Columns` column and convert them to likert numerical scales. Then I compute the Absolute deviation between the ground truth's number and predicted answers's and divide by range of possible answer,similar to being reported by authors as mentioned above.



| Question Type | Metric | Accuracy
|------ | ----- | ------
|Multiple Choice (MC) | Exact Match | 78.07%
|Matrix | Mean Absolute Deviation |83.45%


Blockwise Matrix Evaluation Metrics

| Question Type | Block Name                                             | Mean Normalized Accuracy | Unique Questions | Total Responses | % of Total Responses |
|------------|--------------------------------------------------------|-------------------------:|-----------------:|----------------:|---------------------:|
| Matrix     | False consensus                                        | 87.44%                   | 1                | 2,058           | 20.00%               |
| Matrix     | Non-experimental heuristics and biases                 | 86.32%                   | 2                | 4,116           | 40.00%               |
| Matrix     | Linda -no conjunction                                  | 83.24%                   | 1                | 1,029           | 10.00%               |
| Matrix     | Linda-conjunction                                      | 81.57%                   | 1                | 1,029           | 10.00%               |
| Matrix     | Probability matching vs. maximizing - Problem 2        | 76.17%                   | 1                | 1,026           | 9.97%                |
| Matrix     | Probability matching vs. maximizing - Problem 1        | 73.36%                   | 1                | 1,032           | 10.03%               |

I also didn't use this the same evaluation dataset to measure against the LLM as the model requires training test splits and it won't be correct to split by rows or by question types directly. I discuss in the next section the strategy use to create the train and test/evaluation dataset.

###  Dataset Biases



-----------------

###### Notes



Performance gaps between human performance and LBM 0 shot - 
1. Across categories performance varies - larger gap in pricing,sunk cost fallacy


Next steps
1. Think removing poor questions 



# Section 2 Modelling Strategy


###### Context Handling


- Large context size - Summaries or how to handle full text
- Prompt based compression techniques 
    - LLMLingua
- Base model -base, instruction tuned model ?

##### Sampling Strategy for context window

###### Q&A Arrangenment


###### Compressing techniques
Larger model can compress input prompt and provide the input to the smalle




### Model Techniques


#### Baseline Model
Baseline model will be a 0 shot version of the model we choose.

#### Supervised Fine Tuning (SFT)
Why ? 


#### Base model vs instruction tuned 


#### Objective Function
- **Loss Function** - Cross Entropy Loss 
- **Target Variable** - `wave4_Q_wave4_A` column from `Wave Split` folder
- **Demographic data exclusion for loss computation** - As Demographic Q&A are not a behavioral trait, we should exclude it for optimizing loss, and use only input data for training. To implement, mask these token as -100.

#### Key Hyperparameters



###### Improving SFT Models with RLVR
As the answers have a fixed deterministic range, we can reward the model whose outputs are more closer to actual values compared to whose answer are further away. In a way it will help the SFT model to learn latent 



###### Risk and Mitigations
SFT Fine Tune model stops learning human behavior and memorizes.


### Executing Models

#### Prerequisites
```
# Get Pip
sudo apt update && sudo apt install python3-pip -y

# Install Torch
nvidia-smi

# Cuda CPU
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cuda 12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Cuda 13
pip3 install torch torchvision

# Install Transformers
pip install transformers

# Other packages
pip install accelerate wandb peft trl==0.9.6

# Wandb Set up 
wandb login 
API Key ce18e8ae96d72cd78a7a54de441e9657bc0a913d

```

##### Sampling Technique


##### Current Sampled dataset statstics
- Train 80 personas * 50 Questions = 4000 training rows
- Val 10 personas * 50 Questions = 500 validation rows
- Test 10 personas * All Questions from wave 4 

#### Step 1 Prepare
```
python src/models/data-prep.py
```

#### Step 2 Train Model


##### SFT Fine Tuned Script to run
Train on Wave 1-3 dataset 
```
cd lbm

export MODEL_DIR='qwen3-0.6b-sft'
rm -rf checkpoints/$MODEL_DIR
mkdir checkpoints checkpoints/$MODEL_DIR  checkpoints/$MODEL_DIR/logs

export DATA_DIR='datasets/datasplits'
mkdir datasets/datasplits
mkdir datasets/results datasets/results/$MODEL_DIR

accelerate launch --num_processes 1 src/models/train.py \
    --train_data_path ""$DATA_DIR"/train_10P_5Q.jsonl" \
    --val_data_path "datasets/datasplits/val_1P_5Q.jsonl" \
    --output_dir "checkpoints/$MODEL_DIR" \
    --model_id "Qwen/Qwen3-0.6B" \
    --batch_size 1 \
    --grad_accum 4 \
    --epochs 1 \
    --learning_rate 2e-5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --wandb_project "LBM-twin-models" \
    --wandb_run_name "$MODEL_DIR"


```

##### Step 3 Inference
Inference on Wave 4 dataset of unseen persona

```
cd lbm
MODEL_DIR = 'qwen3-0.6b-sft'
export DATA_DIR='datasets/datasplits'
mkdir datasets/results datasets/results/$MODEL_DIR

python src/models/inference.py \
    --adapter_path "checkpoints/$MODEL_DIR/final_sft_adapter" \
    --test_data_path "$DATA_DIR/test_1P_5Q.jsonl" \
    --output_file "datasets/results/$MODEL_DIR/predictions.jsonl" \
    --batch_size 1
```

#### Step 4 Evaluate Metrics

```
cd lbm
export MODEL_DIR='qwen3-0.6b-sft'
python src/models/evaluate.py --predictions "datasets/results/$MODEL_DIR/predictions.jsonl" --metrics_path datasets/results/$MODEL_DIR/metrics.json
```

**Final Metrics Report**
- File Path - datasets/results/$MODEL_DIR
- File Name - metrics.json


# Section 3 Evaluation
#### Evaluation Metrics 

1. **Accuracy** As we have 2 Question types of Multiple choice (MC) and Matrix questions, 
    we can compute Exact Match Rate and Mean Absolute Deviation accuracy metrics of the digital twin model vs Wave 4 ground truth answers. 
2. **Correlation (For Matrix Type Questions)**
    - For Matrix type questions having likert scales, we can measure pearson correlation.
    - It should be measure for the following sets - 
        - Set 1. The correlation between human's answers in Wave1-3 vs Wave4 answer to check their agreement with themselves.
        - Set 2. The correlation between human's answers in Wave 4 vs Digital Twin model.

3. **Cohen's Kappa (For Multiple Choice Questions)**
    We should also measure the agreement between the following sets  
    - Set 1. The correlation between human's answers in Wave1-3 vs Wave4 answer to check their agreement with themselves.
    - Set 2 . The correlation between human's answers in Wave 4 vs Digital Twin model.
    - Multiple choice questions have 2 types of answers, ordered or unordered.
        - Type 1 : Ordinal Answers - In the example Question we have order in the range from I am so sad or unhappy that I can't stand it -> I don't feel sad. 
           
           Metric :  Cohen's Weighted Kappa
           

         ```
        "QuestionID": "QID126",
        "QuestionText": "",
        "QuestionType": "MC",
        "Options": [
        "I don't feel sad",
        "I feel sad",
        "I am sad all the time and I can't snap out of it",
        "I am so sad or unhappy that I can't stand it"
        ]
        ```
            
        - Type 2 : Unordered Answers
        In the example question, its a yes or no type of question.

          Metric :  Cohen's Kappa
         Range  : -1 to 1, with >0.8 rating considered perfect agreement and <0 considered poor

        ```
         "QuestionID": "QID9_10",
        "QuestionText": "Please consider the following product category: Lunchmeat - Sliced - Refrigerated. Suppose you are in a grocery store, and you see the following product in that category: Oscar Mayer Chopped Ham & Water product Deli Lunch Meat, 16 Oz Package. The product is priced at: $0.87. Would you or would you not purchase this product?",
        "QuestionType": "MC",
        "Options": [
        "Yes, I would purchase the product",
        "No, I would not purchase the product"
        ]
        ```

#### Performance Ceiling  
The test-retest reliability metrics can serve as the ceiling criteria for benchmarking the models. For the 2 question types Multiple Choice and Matrix type question, we can ......

#### Baseline to beat 
A good baseline could be randomly guessing the answers as the base model should be able to beat if it does truly learn something from the dataset.

#### Data Leakage
- **Persona summaries and usage of `Full personal` folder** - 
  -  For the fields `persona_text`, the dataset information states that 
    ```Complete survey responses in text format, including all questions and answers. For questions that appear in both waves 1-3 and wave 4, the wave 4 responses are used.```
    This field contains the wave4 responses. As waves 1-3 is the data to be used for training, and to prevent future data leakage, we can't utilize this field.  
    The corresponding `persona_summary` seems to be generated from the full dataset of all 4 waves combined. If we plan to use the summary of wave 1-3, this field doesn't seem to be reliable.
    We need to generate our own summaries if needed for training from the Wave Split folder's `wave1_3_persona_text` or `wave1_3_persona_json`

- **Target Leakage** - If we split the dataset by rows, we will have situation where for a given person and a question, the training dataset has seen their wave 1-3 answer and will memorize and predict the same answer in wave4 which is from future time frame.

#### Train Test Splits
- To measure generatization, we should split at the `participant_id` level to prevent target leakage. The splits I am going with is splits at 70/10/20 for train,validation and test set respectively.


# Section 4 Business Application 
Text write up
# Section 5 Long run maintenance 

###### Model Artifacts 
Following artifacts should be maintained for measuring model performance,debugging purposes.
- Prompt Registry
- Prediction Tags - Tag every predictin of the model with prompt version,model version ID (multiple model IDs in case of cascaded models like summarizer model id + SFT/RL model id)

###### Measuring model drifts


###### Governance of the model 

###### Model Retrainign Trigger Requirements

Adding a new versioned model can depend upon
- If the psycology metrics of a testing version improves compared to the current production model
- Or
- Current model degrades in performance accuracy



# Section 6 Modelling Codes
# Start SFT plan


###### References
https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

https://numiqo.com/tutorial/cohens-kappa

https://arxiv.org/pdf/2606.05336

https://github.com/microsoft/LLMLingua



#### Todos/ Next Steps
**Training script**
-  Dynamically sample number of personas and questions per persona for train/test/validation sets.

** Context Processing**
1. Before preparing the final prompt, Descriptive blocks should be appended to the corresponding questions and then added to the context.
2. Waves numbers (1,2,3) can be added to separate the set of questions of different waves as they represent different period of time.
