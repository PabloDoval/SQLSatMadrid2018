from azureml.train.hyperdrive import *

# Hyperparameter sampling space
ps = RandomParameterSampling(
     {
         '--learning_rate': uniform(0.000001, 0.1),
         '--dropout': uniform(0.5, 0.95)
     }
)

# Early termination policy
early_termination_policy = BanditPolicy(slack_factor = 0.15, evaluation_interval=10)

# Configure the run
hyperdrive_run_config = HyperDriveRunConfig(estimator = tf_estimator,
                                            hyperparameter_sampling = ps,
                                            policy = early_termination_policy,
                                            primary_metric_name = "Accuracy_test",
                                            primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
                                            max_total_runs = 100,
                                            max_concurrent_runs = 10)

# Run
hd_run = Experiment(ws,'mnist').submit(hyperdrive_run_config)

# Launch the widget to view the progress and results
RunDetails(hd_run).show()