
# Section 1 Dataset Analysis

### Dataset Statistics

###### Token statistics
- Why ? To help decide which model to choose in terms of context length.
- Full Persona - For summary text of a persona profile, maximum of 3,061 words and average of 2,014 words, can be approximated as X tokens based on 1 token = 3/4 word 
- Wave Dataset - Answer are around similar length for wave1-3 and wave 4 and are around 6320 words.

##### Dataset Analysis
- Full Persona - The column -> persona_text seems not usable for training. In dataset definition, the authors has mentioned for group of questions that appear in both waves 1-3 and wave 4, the wave 4 responses are used. This would lead to data leakage of these group of questions if we use this for training.

##### Question Analysis 

 
- Total Questions Answered :  353929 

- Wave 1 to Wave 3 Question Types

| Block Name                      | Unique Question Counts | Question Types          | 
|---------------------------------|-----------------------:|-------------------------|
| Cognitive tests                 | 69                     | DB, MC, TE              |
| Personality                     | 49                     | DB, MC, Matrix, TE      |
| Economic preferences            | 36                     | DB, MC, Matrix, TE      |
| Demographics                    | 14                     | MC                      |
| Economic preferences - intro    | 2                      | DB                      |
| Forward Flow                    | 1                      | TE                      |

- Add number of times this block was answered across its Q types
- Add sample question per block per question type

- Add  percentage contribution column 
Where DB = Descriptive Bloc, MC = Multiple Choice, TE = Text Entry and Matrix = Likert scale

- Wave 4  
- Add table 



- Example Question Types
DB,MC,TE, Matrix


#### Other Findings
- DB questions are descriptive blocks and should not be considered for evaluation.
I removed 2058 rows from the evaluation set . But these block could be useful for training data as they provide LLMs with the context.
- Can users skip the questions ? 
    - To analyze user Settings -> ForceResponse

##### Answer Analysis


### Test-Retest Reliability

Total Question To Evaluate after dropping DB : 129654

Total Question To evaluate after dropping empty in either wave : 119364

|  Question Type   | Number of Questions 
|-----|-----| 
|Multiple Choice (MC) | 109,074 | 
|Matrix | 10,290
|Total | 119,364
As Evaluation will differ by Question Types as they have different answer ranges, we must compute metrics for them individually. As reported in the paper the answer are of 2 types binary or numerical.

For MC Question types, I compute the exact match and then get the % of questions correctly answer by participants.

For Matrix Question types, which we have 36 unique questions. As each question can have different range, I utilized the question_catalog.json file to get those ranges from the `Columns` column and convert them to likert numerical scales. It is then mapped to the user answer and then I computed metric stated by author as
 ```
 For non-binary measures, we calculate the absolute deviation between the ground truth and predicted answer, divided by the range of possible answers.5 We then compute accuracy as 1 minus this absolute deviation. This measure generalizes accuracy from binary to numerical questions: it ranges between 0 and 1, is equal to 1 when the prediction is equal to the ground truth, and 0 when it is maximally different. When multiple questions are included in the same task, we take the mean accuracy across the questions within each task
 ```

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





These human numbers of test-retest accuracy can be used to compare with the LLM performance.

###  Dataset Biases



-----------------

###### Notes



Performance gaps between human performance and LBM 0 shot - 
1. Across categories performance varies - larger gap in pricing,sunk cost fallacy


Next steps
1. Think removing poor questions 


###### References
https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them


# Section 2 Modelling Strategy

###### Train Test Splits
- To measure generatization, we should split at the `participant_id` level
with splits at 70/10/20 for train,validation and test set respectively.
- Data Leakage at question level - Same set of question. Every single one of those 85 questions in Wave 4 is a direct repeat of a question they were already asked in Waves 1-3. The wave 4 question set should be excluded from the training dataset itself.

###### Context Handling


- Large context size - Summaries or how to handle full text
- Prompt based compression techniques 
    - LLMLingua
- Base model -base, instruction tuned model ?

###### Q&A Arrangenment


###### Compressing techniques
Larger model can compress input prompt and provide the input to the smalle




###### Model Techniques


###### Baseline Model
Baseline model will be a 0 shot version of the model we choose.

###### Supervised Fine Tuning (SFT)
Why ? 


###### Base model vs instruction tuned 


###### Objective Function
- Loss Function - Cross Entropy Loss 
- Exclude Demographic data for model trianing - As Demographic Q&A are not a behavioral trait, we should exclude it for optimizing loss, and use only input data for training. How to implement, mask these token as -100

###### Key Hyperparameters



###### A good candidate model - RLVR
As the answers have a fixed deterministic range, we can reward the model whose outputs are more closer to actual values compared to whose answer are further away. In a way it will help the SFT model to learn latent 



###### Risk and Mitigations
SFT Fine Tune model stops learning human behavior and memorizes.





###### Data Pipeline
Input Features

# Section 3 Evaluation
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