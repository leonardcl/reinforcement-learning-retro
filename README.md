# reinforcement-learning-retro
Experiment on Super Mario Bros and Felix The Cat game using DQN and A2C

## How to Run
1. Download this repository
2. Import the ROM by running ``` python3 -m retro.import .```
3. If you wnat to run testing on the agent you can follow this:
    \nSuper Mario Bros - DQN\n
    ```python rl.py --load_model True --model_name mario-ep-700_policy_net,mario-ep-700_target_net```
    Super Mario Bros - A2C
    ```python rl.py --load_model True --model_name mario-ep-500_actor_net,mario-ep-500_critic_net --algo A2C```
    Felix The Cat
    ```python rl.py --game FTC --load_model True --model_name ep-700_policy_net,ep-700_target_net```
    

## Results
The report can seen in the report file: ```report_leonard.pdf```
The experiment result video uploaded on YouTube:
-	Super Mario Bros - DQN: https://youtu.be/0xGgGnw-ZsU
-	Super Mario Bros - A2C: https://youtu.be/nQbJq5mosaQ
-	Felix The Cat - DQN: https://youtu.be/0Xd5TEFk4GU

