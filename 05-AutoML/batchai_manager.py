from azureml.core.compute import BatchAiCompute, ComputeTarget
from azureml.train.estimator import Estimator

class BatchAIManager():

    def __init__(self, workspace):
        self.__workspace = workspace

    def get_or_create(self, compute_name, min_nodes=2, max_nodes=4, vm_size='Standard_D2_v2'):
        if compute_name in self.__workspace.compute_targets():
            compute_target = self.__workspace.compute_targets()[compute_name]
            if compute_target and type(compute_target) is BatchAiCompute:
                print('Found compute target: ' + compute_name)
                return compute_target

        # If not found, create a new one
        print('Creating a new compute target...')
        provisioning_config = BatchAiCompute.provisioning_configuration(vm_size = vm_size,
                                                                        cluster_min_nodes = min_nodes, 
                                                                        cluster_max_nodes = max_nodes,
                                                                        autoscale_enabled=True)

        compute_target = ComputeTarget.create(self.__workspace, compute_name, provisioning_config)
        
        # Can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        
        # For a more detailed view of current BatchAI cluster status, use the 'status' property
        print(compute_target.status.serialize())

        return compute_target

    # Channels is not used
    def get_script_config(self, script_folder, entry_script, script_params, compute_target, channels=None, conda_packages=None, pip_packages=None):
        estimator = Estimator(source_directory=script_folder,
                            entry_script=entry_script,
                            script_params=script_params,
                            conda_packages=conda_packages,
                            pip_packages=pip_packages,
                            compute_target=compute_target)

        return estimator