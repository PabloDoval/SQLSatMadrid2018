import os
import shutil
from azureml.core import Workspace, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from vm_manager import VMManager

if __name__ == '__main__': 
    # Security for the RBAC principal
    client_id = 'e0ae87de-e27b-433a-b7b5-017b0cd0808f'
    client_secret = 'de90f973-5a74-4e1c-a3a1-fa8265ca8ccd'
    tenant_id = '5c384fed-84cc-44a6-b34a-b060bf102a6e'
    servicePrincipalAuth = ServicePrincipalAuthentication(tenant_id, client_id, client_secret)

    # Compute Target
    compute_target_name = "fashionMNISTVM"
    compute_size_vm = "Standard_NC6"
    
    workspace = Workspace.from_config(auth=servicePrincipalAuth)
    compute_manager = VMManager(workspace)
    compute_target = compute_manager.get_or_create(compute_target_name, vm_size=compute_size_vm)

    print('Prepare environment and code')
    script_folder = './training'
    if (not os.path.exists(script_folder)):
        os.makedirs(script_folder)    
    entry_script = 'train.py'
    shutil.copy(entry_script, script_folder)
    shutil.copy('model.py', script_folder)

    script_params = []
    pip_libs = ['numpy', 'azureml-sdk[automl]', 'tensorflow']
    experiment_config = compute_manager.get_script_config(script_folder, entry_script, script_params, compute_target, pip_packages=pip_libs)

    print('Run experiment')
    experiment_name = 'fashionMNIST'
    experiment = Experiment(workspace=workspace, name=experiment_name)
    run = experiment.submit(config=experiment_config)
    run.wait_for_completion(show_output=True)

    # Register model in workspace
    run.register_model(model_name='model', model_path='./output/fashionMNIST')
