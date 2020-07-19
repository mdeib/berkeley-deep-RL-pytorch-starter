1) The code structure for this homeowrk was heavily modified in order to match the structure of the previous three homeworks. 
To this end the PDF does not give the most accurate location instructions but should still be referred to for questions and guidance.
The logging procedure in particular was changed to match the previous assignments.

2) Code:

Code to look at:

- scripts/train_ac_exploration_f18.py
- envs/pointmass.py
- infrastructure/rl_trainer.py (Has been changed for this homework)
- infrastructure/utils.py (Has been changed foir this homework)

Code to fill in as part of HW:

- agents/ac_agent.py (new Exploratory_ACAgent class added)
- exploration/exploration.py
- exploration/density_model.py

3) commands to run can be found in the run.sh file in scripts

4) Visualize saved tensorboard event file:

$ cd cs285/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)