from azureml.core.compute import DsvmCompute, RemoteCompute
from azureml.core.runconfig import RunConfiguration, DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import ScriptRunConfig

class VMManager():

    def __init__(self, workspace):
        self.__workspace = workspace

    def get_or_create(self, compute_name, vm_size):
        compute_target = self.__get_if_exist_compute_target(compute_name)

        if (compute_target is None):
            print('Creating a new compute target...')
            vm_config = DsvmCompute.provisioning_configuration(vm_size=vm_size)
            compute_target = DsvmCompute.create(self.__workspace, name=compute_name, provisioning_configuration=vm_config)
            compute_target.wait_for_completion(show_output = True)

        return compute_target
        
    def attach(self, compute_name, username, password, address):
        compute_target = self.__get_if_exist_compute_target(compute_name)

        if (compute_target is None):
            compute_target = RemoteCompute.attach(self.__workspace, name=compute_name, username=params['username'], address=params['address'], password=params['password'])
            compute_target.wait_for_completion(show_output=True)

        return compute_target

    def get_script_config(self, script_folder, entry_script, script_params, compute_target, channels=None, conda_packages=None, pip_packages=None):
        run_config = self.__get_run_config(compute_target, channels, conda_packages, pip_packages)
        script_run_config = ScriptRunConfig(source_directory=script_folder, script=entry_script, arguments=script_params, run_config=run_config)
        return script_run_config

    def __get_if_exist_compute_target(self, compute_name):
        compute_target = None

        if compute_name in self.__workspace.compute_targets():
            target = self.__workspace.compute_targets()[compute_name]
            if target and (type(target) is DsvmCompute or type(target) is RemoteCompute):
                print('Found compute target: ' + compute_name)
                compute_target = target

        return compute_target
        
    def __get_run_config(self, compute_target, channels=None, conda_packages=None, pip_packages=None):
        # Load the "cpu-dsvm.runconfig" file (created by the above attach operation) in memory
        run_config = RunConfiguration(framework = "python")

        # Set compute target to the Linux DSVM
        run_config.target = compute_target.name

        # Use Docker in the remote VM
        run_config.environment.docker.enabled = False

        # Ask system to provision a new one based on the conda_dependencies.yml file
        run_config.environment.python.user_managed_dependencies = False

        # Prepare the Docker and conda environment automatically when used the first time.
        run_config.auto_prepare_environment = True

        # specify dependencies obj
        conda_dependencies = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)
        if (channels):
            for channel in channels:
                conda_dependencies.add_channel(channel)

        run_config.environment.python.conda_dependencies = conda_dependencies

        return run_config

