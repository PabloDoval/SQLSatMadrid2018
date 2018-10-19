import os
import shutil
from azureml.core import Workspace, Experiment
from training.compute_target.vm_manager import VMManager

if __name__ == '__main__': 
    #az login
    #az account set --subscription <subscriptionId>

    compute_target_name = "fashionMNISTVM"
    compute_size_vm = "Standard_NC6" # "Standard_D2_v2"
    
    workspace = Workspace.from_config()
    compute_manager = VMManager(workspace)
    compute_target = compute_manager.get_or_create(compute_target_name, vm_size=compute_size_vm)

    # print('Load data')
    # data_store = workspace.get_default_datastore()
    # data_store.upload(src_dir=config['train']['local_dataset_folder'], target_path=config['train']['remote_dataset_folder'], overwrite=True, show_progress=True)

    print('Prepare environment and code')
    script_folder = './training'
    entry_script = 'train.py'
    shutil.copy(entry_script, script_folder)
    shutil.copy('model.py', script_folder)

    script_params = []
    conda_libs = ['numpy', 'tensorflow', 'azureml-sdk[automl]']
    experiment_config = compute_manager.get_script_config(script_folder, entry_script, script_params, compute_target, conda_packages=conda_libs)

    print('Run experiment')
    experiment_name = 'fashionMNIST'
    experiment = Experiment(workspace=workspace, name=experiment_name)
    run = experiment.submit(config=experiment_config)
    run.wait_for_completion(show_output=True)

    # Register model in workspace
    run.register_model(model_name='model', model_path='./output/fashionMNIST')
