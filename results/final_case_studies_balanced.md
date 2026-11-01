# Balanced qualitative examples (A vs. D)

Four balanced illustrative cases selected from the paired 300-turn results: one correction, one regression, one case where both are incorrect, and one where both are correct. These are examples, not an independent evaluation sample.

## Case 1: D corrects a baseline error
- **Turn:** `conv000_Fear_of_layoff_from__18`
- **Situation:** Fear of layoff from job
- **Last seeker utterance:** That is it my friend! A lot of people are losing their jobs on daily basis, that one a lone is scaring
- **Gold strategy:** Others
- **A (baseline):** Affirmation and Reassurance
- **D (full):** Others
- **A response:** Scary, but you've got this. Keep applying and stay positive.
- **D response:** Many are in the same boat; focus on your strengths and start networking.
- **D contrastive cues:** Note: 'Affirmation and Reassurance' and 'Providing Suggestions' can look similar. 'Affirmation and Reassurance' means: Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement. 'Providing Suggestions' means: Provide suggestions about how to change, but be careful to not overstep and tell them what to do. Choose based on which function the situation actually calls for. Note: 'Affirmation and Reassurance' and 'Others' can look similar. 'Affirmation and Reassurance' means: Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement. 'Others' means: Exchange pleasantries and use other support strategies that do not fall into the above categories. Choose based on which function the situation actually calls for.

## Case 2: D changes a correct baseline prediction to an error
- **Turn:** `conv004_The_pandemic_has_pla_15`
- **Situation:** The pandemic has placed me in a financial crisis and it has negatively impacted my academic standing.
- **Last seeker utterance:** What kind of jobs did you take?
- **Gold strategy:** Providing Suggestions
- **A (baseline):** Providing Suggestions
- **D (full):** Information
- **A response:** Try tutoring or part-time academic support roles.
- **D response:** I took part-time tutoring and delivery jobs to help.
- **D contrastive cues:** Note: 'Affirmation and Reassurance' and 'Providing Suggestions' can look similar. 'Affirmation and Reassurance' means: Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement. 'Providing Suggestions' means: Provide suggestions about how to change, but be careful to not overstep and tell them what to do. Choose based on which function the situation actually calls for.

## Case 3: Both configurations are incorrect
- **Turn:** `conv000_Fear_of_layoff_from__6`
- **Situation:** Fear of layoff from job
- **Last seeker utterance:** Not about feeling well, it is about layoff from my job
- **Gold strategy:** Others
- **A (baseline):** Reflection of feelings
- **D (full):** Question
- **A response:** I'm sorry to hear that. Losing a job can be really tough.
- **D response:** Can you tell me more about how the job layoff is affecting you?
- **D contrastive cues:** Note: 'Affirmation and Reassurance' and 'Others' can look similar. 'Affirmation and Reassurance' means: Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement. 'Others' means: Exchange pleasantries and use other support strategies that do not fall into the above categories. Choose based on which function the situation actually calls for. Note: 'Affirmation and Reassurance' and 'Reflection of feelings' can look similar. 'Affirmation and Reassurance' means: Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement. 'Reflection of feelings' means: Articulate and describe the user's feelings. Choose based on which function the situation actually calls for.

## Case 4: Both configurations are correct
- **Turn:** `conv014_My_kids_are_driving__9`
- **Situation:** My kids are driving me crazy now that we are all having to work and school our kids from home. My husband and I both work from home now because of Corona. My 15 year old boy and 8 year old girl argue after they finish schooling and my husband and I are still working. It is starting to interrupt the quality of our work, Any suggestions? They are going crazy since the emergence of the Corona Virus.
- **Last seeker utterance:** Talking is great when we are doing it and they all agree. But it never stays this way.
- **Gold strategy:** Providing Suggestions
- **A (baseline):** Providing Suggestions
- **D (full):** Providing Suggestions
- **A response:** You could establish a quiet signal and discuss its importance as a family.
- **D response:** Try setting a calm time for discussions and use a timer for quiet periods.
- **D contrastive cues:** Note: 'Others' and 'Providing Suggestions' can look similar. 'Others' means: Exchange pleasantries and use other support strategies that do not fall into the above categories. 'Providing Suggestions' means: Provide suggestions about how to change, but be careful to not overstep and tell them what to do. Choose based on which function the situation actually calls for.
