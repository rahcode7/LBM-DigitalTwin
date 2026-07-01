

### Dataset Statistics

###### Token statistics
- Why ? To help decide which model to choose in terms of context length.
- Full Persona - For summary text of a persona profile, maximum of 3,061 words and average of 2,014 words, can be approximated as X tokens based on 1 token = 3/4 word 
- Wave Dataset - Answer are around similar length for wave1-3 and wave 4 and are around 6320 words.

###### Dataset Analysis
- Full Persona - The column -> persona_text seems not usable for training. In dataset definition, the authors has mentioned for group of questions that appear in both waves 1-3 and wave 4, the wave 4 responses are used. This would lead to data leakage of these group of questions if we use this for training.


###### Notes
To check how often the user changed answer from wave 1-3 to wave 4 



Performance gaps between human performance and LBM 0 shot - 
1. Across categories performance varies - larger gap in pricing,sunk cost fallacy


Next steps
1. Think removing poor questions 