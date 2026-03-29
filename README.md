# fpl-bot
Making FPL bot using PyTorch

Data ingestion is fairly messy due to:
                                - DEFCON points being added in 25/26
                                - FPL API data incomplete for 24/25 from FPL Core insight,
                                  so suplemented with vaastav data
                                - varying data quality between the two providers

Currently DEFCON calculation from last season off by ~5% due to inconsitencies between FPL and Opta stats.
Currently investigating DEFCON also being wrong for this season due do inaccuracies in FPL-Core-Insight data.

Model currently works predicting points for one week. Will look to predict a target window, points in a range of future gameweeks.

The model is fairly basic and currently tuned to perform optimally, but probably held back by data limitations, predicts points on validatioon group with mae ~ 1 point.

I have also currently maxed out my limited deep learning knowledge, so will do some research into improving the model.

Final Goal: - Implement a basic AI agent to use models predictions to play a season of fpl, via RL.

Data from:
        - https://github.com/olbauday/FPL-Core-Insights
        - https://github.com/vaastav/Fantasy-Premier-League
        
