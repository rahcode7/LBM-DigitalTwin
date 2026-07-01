
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
|Matrix | 10,290 |
|Total | 119,364 |


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


# Section 3 Evaluation
# Section 4 Business Application 
Text write up
# Section 5 Long run maintenance 
Text write up 
# Section 6 Modelling Codes
SFT etc code to write
