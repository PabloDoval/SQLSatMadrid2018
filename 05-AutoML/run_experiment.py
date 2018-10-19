import os
import shutil
from azureml.core import Workspace, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from batchai_manager import BatchAIManager

if __name__ == '__main__': 
    # Security for the RBAC principal
    client_id = 'e0ae87de-e27b-433a-b7b5-017b0cd0808f'
    client_secret = 'de90f973-5a74-4e1c-a3a1-fa8265ca8ccd'
    tenant_id = '5c384fed-84cc-44a6-b34a-b060bf102a6e'
    servicePrincipalAuth = ServicePrincipalAuthentication(tenant_id, client_id, client_secret)

    # Compute Target
    compute_target_name = "fashionMNISTBAI"

    workspace = Workspace.from_config(auth=servicePrincipalAuth)
    compute_manager = BatchAIManager(workspace)
    compute_target = compute_manager.get_or_create(compute_target_name)

    print('Prepare environment and code')
    script_folder = './training'
    shutil.copy('get_data.py', script_folder)
        
    automl_settings = {
        "max_time_sec": 120,
        "iterations": 20,
        "n_cross_validations": 5,
        "primary_metric": 'AUC_weighted',
        "preprocess": False,
        "concurrent_iterations": 5,
        "verbosity": logging.INFO
    }

    automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             path = './',
                             compute_target = compute_target,
                             data_script = script_folder + '/get_data.py',
                             **automl_settings
                            )

    experiment = Experiment(workspace=workspace, name='fashionMNIST_autoML')
    remote_run = experiment.submit(automl_config,   show_output=False)
    