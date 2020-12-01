"""
"run_AHIRL.py" executes the training schedule for the agent. 
"""

import pickle as cpickle
import agent as Agent
from utils import print_summary


TEST_FREQ = 2

num_test_episodes = 1000

def run_AHIRL(FLAGS,env,agent):

    # Print task summary
    print_summary(FLAGS,env)

    total_episodes = 0

    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    print('FLAGS.test', FLAGS.test)
    print('FLAGS.train_only', FLAGS.train_only)
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True

    num_episodes = agent.other_params["num_exploration_episodes"]
    model_save_freq = 10

    if FLAGS.results:
        NUM_BATCH = int(num_episodes / model_save_freq)
    else:
        NUM_BATCH = 1

    suc_rate = []
    attempt_avg = []
    att_sum = []
    att = 0

    for batch in range(NUM_BATCH):
        test_suc = 0
        attempt_sum = 0
        attempts_vec = []

        agent.initialize_networks(batch)

        if FLAGS.test:
            num_episodes = 500

        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            num_episodes = num_test_episodes

            # Reset successful episode counter
            successful_episodes = 0

        for episode in range(num_episodes):

            print("\nBatch %d, Episode %d" % (batch, episode))

            # Train for an episode
            success = agent.train(env, episode, total_episodes)

            if FLAGS.show:
                while agent.worker.reset:
                    agent.worker.unsetReset()
                    success = agent.train(env, episode, total_episodes)
            attempts_vec.append(agent.attempts)

            if FLAGS.test and FLAGS.results:
                attempt_sum += agent.attempts

            if success:
                print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))

                # Increment successful episode counter if applicable
                if mix_train_test and batch % TEST_FREQ == 0:
                    successful_episodes += 1
                if FLAGS.test and not mix_train_test:
                    test_suc += 1

            if FLAGS.train_only or (mix_train_test and batch % TEST_FREQ != 0):
                total_episodes += 1

            if total_episodes % model_save_freq == 0 and total_episodes > 0:
                # Save agent
                agent.save_model(total_episodes)

            file_name1 = "attempts.pkl"
            if FLAGS.active_learning == 1:
                file_name1 = "al_noise_"+file_name1
            if FLAGS.active_learning == 2:
                file_name1 = "al_bag_"+file_name1
            if FLAGS.al_test == 1:
                file_name1 = "al_noise_"+file_name1
            if FLAGS.al_test == 2:
                file_name1 = "al_bag_"+file_name1            
            file_name1 = "exp%d_"%(FLAGS.expnum) + file_name1
            if not FLAGS.test:
                with open("models/"+file_name1, 'wb') as f:
                    cpickle.dump(attempts_vec, f)

        if FLAGS.test and FLAGS.results:
            suc_rate.append(test_suc / num_episodes)
            file_name2 = "success_rate.pkl"
            if FLAGS.al_test == 1:
                file_name2 = "al_noise_"+file_name2
            if FLAGS.al_test == 2:
                file_name2 = "al_bag_"+file_name2
            file_name2 = "exp%d_"%(FLAGS.expnum) + file_name2
            with open("models/"+file_name2, 'wb') as f:
                cpickle.dump(suc_rate, f)

            # Average attempts in each batch (model) during test 
            attempt_avg.append(attempt_sum / num_episodes)
            file_name3 = "attempt_avg.pkl"
            if FLAGS.al_test == 1:
                file_name3 = "al_noise_"+file_name3
            if FLAGS.al_test == 2:
                file_name3 = "al_bag_"+file_name3
            file_name3 = "exp%d_"%(FLAGS.expnum) + file_name3
            with open("models/"+file_name3, 'wb') as f:
                cpickle.dump(attempt_avg, f)

            attempts = cpickle.load(open("models/"+file_name1, 'rb'))
            
            for i in range(batch*10, batch*10 + 10):
                att += attempts[i]
            att_sum.append(att)


            file_name4 = "expert_cost.pkl"
            if FLAGS.al_test == 1:
                file_name4 = "al_noise_"+file_name4
            if FLAGS.al_test == 2:
                file_name4 = "al_bag_"+file_name4
            file_name4 = "exp%d_"%(FLAGS.expnum) + file_name4
            with open("models/"+file_name4, 'wb') as f:
                cpickle.dump([att_sum, suc_rate], f)            

        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:

            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            agent.log_performance(success_rate)
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")
        print("Test succ rate: %.2f" % (test_suc / (num_episodes)))
