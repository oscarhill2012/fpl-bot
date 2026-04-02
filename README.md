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

Files that look to try and predict scoring metrics (goals, assists, ...) and then use FPL deterministic scoring rules exist with prefix "multihead". I investigated using these btu currently model just hacks predicting these metrics as is still only evaluated on points loss function. Will look to add loss function for predicting individual metrics. 

Points prediction works fine with MAE ~ 1 point, predictions 0-6 points. As expected, unable to predict any 8+ point hauls.

I was hoping to incorperate a fixture prediction model, i.e predict score and clean sheet percentage for each fixture from strengths and elos. This would have been a seperate model that would have been trained and frozen. I have put this on hold however since there is just not enough consistent data to train this.

This would have been cruicial to trying to implement the model as a data simulator to train an RL agent, since the data simulation would need Liverpool get CS, all liverpool defenders get CS points. The massive inconsistencies between FPL-Core-Insights 24/25 and 25/26 data means there isnt really anything to train the fixture model on. This is a shame, as it means there isnt really a practical way to simulate match dependancy for players and since the data for FPL with DEFCON points is essentially limited to the current season, I don't see an easy way to solve the data scarcity.

Due to this I am going to stop working on this bot, I may come back to try and improve points prediction by implementing loss functions for scoring features.  

Data from:
        - https://github.com/olbauday/FPL-Core-Insights
        - https://github.com/vaastav/Fantasy-Premier-League
        
