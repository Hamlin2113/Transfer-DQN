A transfer learning algorithm for cognitive radio

Introduction:

This project provides transfer DQN for cognitive radio systems. Inspired by experience replay techniques, a dual experience pool strategy was adopted, which stores transitions from both the source and target domains. By adjusting the weights, the impact of the samples during the environment transition is controlled, thereby preventing negative transfer issues.

The algorithms include; 

Transfer-Deep Q-Networks
Comparison algorithms
Spectrum environments for both FHSS scenario, sweep jamming scenario, and Markov scenario

Usage for case x:
locate: 
run: /Dynamic_spectrum_Access_simulation/Main/

step 1.
step0_generate_Markov2JumpTable.py
step 2.
step1_1store_memory_for_jump_table.py
step 3. 
step1_1store_memory_for_jump_table_case x.py
step 4. 
step2_1_transfer_test_experiment2.py
step 5. 
step3_dynamic_figure.py


Credits
The implementation of Transfer-DQN is based on the paper "Transfer Reinforcement Learning for Dynamic Spectrum Environment" by Sheng et al.
