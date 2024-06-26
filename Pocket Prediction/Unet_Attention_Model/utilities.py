import subprocess
import re

# function to list the GPU devices that we have access to and are going to run this program on
def count_gpus():
    
	# execute the 'nvidia-smi' command to get GPU details
    nvidia_smi_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout

    # use regular expression to count how many times 'MiB /' appears, indicating GPU memory usage
    gpu_count = len(re.findall(r'MiB /', nvidia_smi_output))

    return gpu_count

