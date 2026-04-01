# fpl-bot
Making FPL bot using PyTorch

Data ingestion is fairly messy due to:
                                - DEFCON points being added in 25/26
                                - FPL API data incomplete for 24/25 from FPL Core insight,
                                  so suplemented with vaastav data
                                - varying data quality between the two providers

Currently DEFCON calculation from last season off by ~5% due to inconsitencies between FPL and Opta stats.
Currently investigating DEFCON also being wrong for this season due do inaccuracies in FPL-Core-Insight data.

Priors aims to ease cold start limitations of FPL datasets, it's only real purpose is to provide some early season context for an agent to generate a first team from...

There is the option to run a single head (SH) or multihead (MH) predictions. Multihead forces the model to predict features which contribute to points, it is currently very weak though, probably due to loss evalution still being calculated on points, not the individual feature predictions. Im guessing model just hacks out the same points prediction with the simplelest weight, I will look to fix this.

Singlehead, just predicting points for given week, gets MAE ~ 1 point. Prediction is okay, the distribution is fine. Definitely limited by vast amount of 0 points and then the odd big haul 8+ points that seem to come from nowhere. Model does seem to do a reasonable job below 6 points though.

Currently: - adding target loading, so can add loss functions for scoring features in multihead
           - adding fixture prediction model, will add fixture context via a score predictor (this will be helpful when simulating data for RL as we want to be able to model team dynamics, i.e if we predict liverpool to get cs all liverpool defenders need cs points)

Final Goal: - Implement a basic AI agent to use models predictions to play a season of fpl, via RL.

Data from:
        - https://github.com/olbauday/FPL-Core-Insights
        - https://github.com/vaastav/Fantasy-Premier-League
        
