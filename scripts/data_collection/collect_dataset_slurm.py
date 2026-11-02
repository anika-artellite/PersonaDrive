"""
Generates a dataset for training on a SLURM cluster.
Each route file is parallelized on its own machine.
Monitors the data collection and continues crashed processes.
Best run inside a tmux terminal.

OVERVIEW:
This script automates the collection of CARLA simulation datasets by:
1. Finding all route XML files in the data directory
2. Creating and submitting SLURM jobs for each route
3. Monitoring job execution and handling failures
4. Automatically resubmitting failed or incomplete jobs

The glob module is used extensively to find files matching patterns:
- glob.glob(pattern, recursive=True) finds all files matching the pattern
- Supports wildcards: * (any characters), ** (any subdirectories), ? (single char)
- Example: 'data/**/*.xml' finds all XML files in data/ and all subdirectories
"""

from datetime import datetime
import os
import subprocess
import time
import glob  # Used for finding files matching patterns (wildcards like *.xml, **/Town*)
import json
from pathlib import Path
import random
import re


def make_bash(code_dir, route_file_number, agent_name, route_file, ckeckpoint_endpoint, save_pth, seed, carla_root, town, repetition):
    """
    Creates a bash script for running a CARLA simulation route.
    
    This function generates a bash script that:
    1. Sets up all necessary environment variables (CARLA paths, Python paths, etc.)
    2. Starts the CARLA server with specific ports and rendering settings
    3. Waits for the server to initialize (180 seconds)
    4. Runs the leaderboard evaluator to execute the simulation route
    
    Args:
        code_dir (str): Root directory of the simlingo code
        route_file_number (str): Identifier number for the route file
        agent_name (str): Path to the agent Python file that controls the vehicle
        route_file (str): Path to the XML file containing the route definition
        ckeckpoint_endpoint (str): Path where the checkpoint/results will be saved
        save_pth (str): Directory path where simulation data will be saved
        seed (int): Random seed for the traffic manager to ensure reproducibility
        carla_root (str): Root directory of the CARLA installation
        town (str): CARLA town name (e.g., 'Town01', 'Town02')
        repetition (int): Repetition number for this route (for multiple runs)
    
    Returns:
        str: Path to the created bash script file
    """
    save_slurm = save_pth.replace("data/", "slurm/")

    jobfile = f"{save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh"
    # create folder
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)
    run_command = "python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${FREE_WORLD_PORT} \
        --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS} \
            --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} \
                --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=600"

    qsub_template = f"""#!/bin/bash
export SCENARIO_RUNNER_ROOT={code_dir}/scenario_runner_autopilot
export LEADERBOARD_ROOT={code_dir}/leaderboard_autopilot

# carla
export CARLA_ROOT={carla_root}
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:leaderboard_autopilot
export PYTHONPATH=$PYTHONPATH:scenario_runner_autopilot
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={agent_name}
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_file}
export TOWN={town}
export REPETITION={repetition}
export TM_SEED={seed}

export CHECKPOINT_ENDPOINT={ckeckpoint_endpoint}
export TEAM_CONFIG={route_file}
export RESUME=1
export DATAGEN=1
export SAVE_PATH={save_pth}

echo "Start python"

export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

bash {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

sleep 180

{run_command}    
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile

def get_running_jobs(jobname, user_name):
    """
    Queries the SLURM scheduler to get information about currently running jobs.
    
    This function uses the 'squeue' command to retrieve all jobs matching the specified
    job name and username. It parses the output to extract job IDs, route file numbers,
    and process IDs.
    
    Args:
        jobname (str): The name pattern of jobs to search for (e.g., 'collect')
        user_name (str): Username whose jobs should be queried
    
    Returns:
        tuple: A 3-tuple containing:
            - currently_num_running_jobs (int): Total number of matching running jobs
            - routefile_number_list (list[str]): List of route file identifiers from job names
            - pid_list (list[str]): List of SLURM job IDs (process IDs)
    
    Example:
        If a job is named "collect_4170_0", the routefile_number will be "4170_0"
    """
    job_list = subprocess.check_output(
            (
                f"SQUEUE_FORMAT2='jobid:10,username:{len(username)},name:130' squeue --sort V | grep {user_name} | \
                    grep {jobname} || true"
            ),
            shell=True,
        ).decode("utf-8").splitlines()
    currently_num_running_jobs = len(job_list)
    #  line is sth like "4767364   gwb791 eval_julian_4170_0   "
    routefile_number_list = [line.split("_")[-2] + "_" + line.split("_")[-1].strip() for line in job_list]
    pid_list = [line.split(" ")[0] for line in job_list]
    return currently_num_running_jobs, routefile_number_list, pid_list

def get_last_line_from_file(filepath):
    """
    Efficiently reads the last line of a file without loading the entire file into memory.
    
    This function is used to check log files for error messages. It works by:
    1. Opening the file in binary mode
    2. Seeking to the end of the file
    3. Reading backwards until a newline character is found
    4. Reading that last line and decoding it to a string
    
    Args:
        filepath (str): Path to the file to read
    
    Returns:
        str: The last line of the file, or an empty string if the file cannot be read
    
    Note:
        This is particularly useful for large log files where reading the entire
        file would be inefficient. Handles OSError (e.g., empty files) gracefully.
    """
    try:
        with open(filepath, "rb", encoding="utf-8") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
    except:
        last_line=""
    return last_line

def cancel_jobs_with_err_in_log(logroot, jobname, user_name):
    """
    Monitors log files of running jobs and terminates jobs that have encountered fatal errors.
    
    This function checks the last line of each job's log file for specific error patterns
    that indicate the simulation has crashed or become stuck. If such errors are detected,
    the corresponding SLURM job is cancelled using 'scancel'.
    
    Detected error patterns:
    - "Actor ... not found!": CARLA actor reference errors
    - "Watchdog exception - Timeout": Job has exceeded time limits
    - "Engine crash handling finished; re-raising signal 11": Unreal Engine crashes
    
    Args:
        logroot (str): Root directory where log files are stored
        jobname (str): Name pattern of jobs to monitor
        user_name (str): Username whose jobs should be checked
    
    Side Effects:
        - Reads log files from disk
        - Terminates SLURM jobs using 'scancel' command
        - Prints messages to console when jobs are terminated
    """
    # check if the log file contains certain error messages, then terminate the job
    print("Checking logs for errors...")
    _, routefile_number_list, pid_list = get_running_jobs(jobname, user_name)
    for i, rf_num in enumerate(routefile_number_list):
        logfile_path = os.path.join(logroot, f"run_files/logs/qsub_out{rf_num}.log")
        last_line = get_last_line_from_file(logfile_path)
        terminate = False
        if "Actor" in last_line and "not found!" in last_line:
            terminate = True
        if "Watchdog exception - Timeout" in last_line:
            terminate = True
        if "Engine crash handling finished; re-raising signal 11" in last_line:
            terminate = True
        if terminate:
            print(f"Terminating route {rf_num} with pid {pid_list[i]} due to error in logfile.")
            subprocess.check_output(f"scancel {pid_list[i]}", shell=True)

def wait_for_jobs_to_finish(logroot, jobname, user_name, max_n_parallel_jobs):
    """
    Blocks execution until the number of running jobs drops below the maximum threshold.
    
    This function continuously polls the SLURM scheduler to check how many jobs are running.
    It waits until there are fewer than max_n_parallel_jobs running before returning.
    Every 4th iteration (counter % 4 == 0), it also checks for and cancels jobs with errors.
    
    Args:
        logroot (str): Root directory for log files (passed to error checking function)
        jobname (str): Name pattern of jobs to monitor
        user_name (str): Username whose jobs should be counted
        max_n_parallel_jobs (int): Maximum number of parallel jobs allowed
    
    Side Effects:
        - Blocks execution (busy-wait loop with 5 second sleep intervals)
        - Periodically calls cancel_jobs_with_err_in_log to terminate failed jobs
        - Prints status updates to console
    """
    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    print(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    counter = 0
    while currently_running_jobs >= max_n_parallel_jobs:
        if counter == 0:
            cancel_jobs_with_err_in_log(logroot, jobname, user_name)
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
        counter = (counter + 1) % 4

def get_num_jobs(job_name, username):
    """
    Gets the current number of running jobs and the maximum allowed parallel jobs.
    
    This function queries SLURM for the number of jobs currently running that match
    the given job name and username. It also reads the maximum number of parallel jobs
    from a configuration file 'max_num_jobs.txt'.
    
    Args:
        job_name (str): Name pattern of jobs to count
        username (str): Username whose jobs should be counted
    
    Returns:
        tuple: A 2-tuple containing:
            - num_running_jobs (int): Number of currently running jobs
            - max_num_parallel_jobs (int): Maximum allowed parallel jobs (from config file)
    
    Note:
        If 'max_num_jobs.txt' cannot be read, defaults to 1 parallel job.
        This allows dynamic control of parallelism without code changes.
    """
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        ).decode('utf-8').replace('\n', ''))

    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'max_num_jobs.txt'), 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except:
        max_num_parallel_jobs = 1

    return num_running_jobs, max_num_parallel_jobs

def get_which_partition(default):
    """
    Determines which SLURM partition to use for job submission.
    
    Reads the partition name from a configuration file 'partition.txt'. If the file
    doesn't exist or contains an invalid partition name, falls back to the default.
    
    Valid partitions are:
    - a100-galvani: A100 GPU partition
    - 2080-galvani: RTX 2080 GPU partition
    - 2080-preemptable-galvani: Preemptable RTX 2080 partition (lower priority)
    - a100-preemptable-galvani: Preemptable A100 partition (lower priority)
    
    Args:
        default (str): Default partition name to use as fallback
    
    Returns:
        str: The partition name to use for SLURM job submission
    
    Note:
        This allows dynamic partition selection without modifying code,
        useful for adapting to cluster availability.
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'partition.txt'), 'r', encoding='utf-8') as f:
            partition_name = f.read()
            if partition_name not in ["a100-galvani", "2080-galvani", "2080-preemptable-galvani", "a100-preemptable-galvani"]:
                partition_name = default
    except:
        print("partition.txt not found. Using default partition.")
        partition_name = default

    return partition_name



def make_jobsub_file(save_path_data, jobname, route_file_number, partition_name, repetition, timeout="0-02:00"):
    """
    Creates a SLURM job submission script with all necessary resource specifications.
    
    This function generates a complete SLURM batch script that:
    1. Specifies resource requirements (GPUs, CPUs, memory, time limit)
    2. Sets up logging directories and files
    3. Prints diagnostic information (node name, job ID, git branch/commit)
    4. Dynamically allocates free network ports for CARLA (streaming, world, traffic manager)
    5. Calls the bash script created by make_bash() with the allocated ports
    
    Args:
        save_path_data (str): Path where simulation data will be saved
        jobname (str): Name for the SLURM job
        route_file_number (str): Identifier for the route file
        partition_name (str): SLURM partition to submit the job to
        repetition (int): Repetition number for this route
        timeout (str, optional): Maximum job runtime in format "D-HH:MM". Defaults to "0-02:00" (2 hours)
    
    Returns:
        str: Path to the created SLURM job submission script
    
    Resource Requirements:
        - 1 node, 1 task per node
        - 8 CPUs per task
        - 40 GB RAM
        - 1 GPU
    
    Note:
        The script uses 'comm' to find free ports in specific ranges:
        - Streaming: 10000-10400
        - World: 20000-20400
        - Traffic Manager: 30000-30400
    """
    save_slurm = save_path_data.replace("data/", "slurm/")
    os.makedirs(f"{save_slurm}/run_files/logs", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/job_files", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/start_files", exist_ok=True)
    jobfile = f"{save_slurm}/run_files/job_files/{route_file_number}_Rep{repetition}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={jobname}_{route_file_number}
#SBATCH --partition={partition_name}
#SBATCH -o {save_slurm}/run_files/logs/qsub_out{route_file_number}.log
#SBATCH -e {save_slurm}/run_files/logs/qsub_err{route_file_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time={timeout}
#SBATCH --gres=gpu:1
# -------------------------------

echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
scontrol show job $SLURM_JOB_ID

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo "Current branch:"
git branch
echo "Current commit:"
git log -1
echo "Current hash:"
git rev-parse HEAD


export FREE_STREAMING_PORT=`comm -23 <(seq 10000 10400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export FREE_WORLD_PORT=`comm -23 <(seq 20000 20400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export TM_PORT=`comm -23 <(seq 30000 30400 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

sleep 2

echo "start python"
pwd
bash {save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh $FREE_STREAMING_PORT $FREE_WORLD_PORT $TM_PORT
"""
    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile

if __name__ == "__main__":
    """
    Main execution block for automated CARLA dataset collection on a SLURM cluster.
    
    This script orchestrates the entire dataset collection process:
    
    PHASE 1 - INITIALIZATION:
    - Configures paths, job settings, and finds all route XML files
    - Supports both balanced routes and leaderboard (lb1) routes
    - Shuffles routes with fixed seed for reproducibility
    
    PHASE 2 - JOB SUBMISSION:
    - Iterates through all routes and repetitions
    - For each route:
      * Extracts town information from filename or route type
      * Creates bash and SLURM submission scripts
      * Waits for job slot availability
      * Submits job to SLURM and tracks it in meta_jobs dictionary
    - Uses traffic manager seeds to ensure different traffic patterns per run
    
    PHASE 3 - MONITORING AND RECOVERY:
    - Continuously monitors all submitted jobs
    - Checks for errors in log files and cancels problematic jobs
    - Automatically resubmits failed or incomplete jobs (up to 3 attempts)
    - Job is considered complete when:
      * Result file exists
      * Progress indicators show completion (progress[0] >= progress[1])
      * Route score is non-zero
      * Status is not "Failed"
    - Preserves log files from failed attempts for debugging
    - Continues until all jobs are successfully completed
    
    Configuration:
    - Uses 'max_num_jobs.txt' to control parallel job limit
    - Uses 'partition.txt' to select compute partition
    - Routes are found using glob patterns in the route_folder
    
    Data Structures:
    - meta_jobs: dict mapping job_id -> (finished, job_file, result_file, resubmit_count)
    """
    repetitions = 1
    repetition_start = 0
    default_partition = "YOUR_PARTITION" 
    job_name = "collect"
    username = "YOUR_USER"
    code_root = r"/path/to/simlingo"
    carla_root = "/path/to/CARLA/root"
    date = datetime.today().strftime("%Y_%m_%d")
    dataset_name = "simlingo_v2_" + date
    root_folder = r"database/"  # With ending slash
    data_save_directory = root_folder + dataset_name
    log_root = f"{data_save_directory}/slurm"

    route_folder = f"{code_root}/data/simlingo"

    # Find all .xml route files using glob patterns:
    # - **/  searches recursively through all subdirectories
    # - *balanced*/*.xml finds XML files in directories containing 'balanced'
    # - *lb1*/**/*.xml finds XML files anywhere under directories containing 'lb1'
    routes = glob.glob(f"{route_folder}/**/*balanced*/*.xml", recursive=True)
    routes_lb1 = glob.glob(f"{route_folder}/**/*lb1*/**/*.xml", recursive=True)

    routes = routes + routes_lb1

    port_offset = 0
    job_number = 1
    # meta_jobs: Dictionary tracking all submitted jobs
    # Key: SLURM job_id (string)
    # Value: tuple (job_finished, job_file, result_file, resubmitted)
    #   - job_finished (bool): Whether job completed successfully
    #   - job_file (str): Path to SLURM job submission script
    #   - result_file (str): Path to expected results JSON file
    #   - resubmitted (int): Number of times job has been resubmitted (max 3)
    meta_jobs = {}

    #shuffle routes
    random.seed(42)
    random.shuffle(routes)
    seed_counter = 1000000 * repetition_start - 1  # for the traffic manager, which is incremented so that we get different traffic each time

    num_routes = len(routes)
    for repetition in range(repetition_start, repetitions):
        for route in routes:
            seed_counter += 1

            # Extract town name (e.g., 'Town01', 'Town10') from route path using regex
            # If not found in path, infer from route type ('validation' -> Town13, 'training' -> Town12)
            try:
                town = re.search('Town(\\d+)', route).group(0)
            except:
                if 'validation' in route:
                    town = 'Town13'
                elif 'training' in route:
                    town = 'Town12'
                else:
                    print(f"Town not found in route {route}")
                    continue

            # Extract scenario information and route identifiers from file path
            # Example path: .../data/simlingo/lb1_split/routes_training/ControlLoss/Town10/route_22_0.xml
            # scenario_type would be: "lb1_split/routes_training/ControlLoss/Town10"
            # routefile_number would be: "route_22_0" (used as unique identifier)
            scenario_type = route.split("/")[-5:-1]
            scenario_type = "/".join(scenario_type)
            routefile_number = route.split("/")[-1].split(".")[0]  # this is the number in the xml file name, e.g. 22_0.xml
            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"

            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            agent = f"{code_root}/team_code/data_agent.py"

            partition_name = get_which_partition(default_partition)

            # Create bash script and SLURM job file
            bash_file = make_bash(code_root, routefile_number, agent, route,
                                  ckpt_endpoint, save_path, seed_counter, carla_root, town, repetition)
            job_file = make_jobsub_file(save_path, job_name, routefile_number, partition_name, repetition, "0-04:00")

            # Wait until submitting new jobs that the #jobs are at below max
            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
            print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
            while num_running_jobs >= max_num_parallel_jobs:
                num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
                time.sleep(0.05)

            print(f"Submitting job {job_number}/{num_routes}: {job_name}_{routefile_number}. ", end="")
            time.sleep(1)
            # Submit job to SLURM and extract the job ID from output (format: "Submitted batch job 12345")
            jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8") \
                                                            .strip().rsplit(" ", maxsplit=1)[-1]
            print(f"Jobid: {jobid}")
            # Track job: Initially not finished, 0 resubmissions
            meta_jobs[jobid] = (False, job_file, ckpt_endpoint, 0)  # job_finished, job_file, result_file, resubmitted
            job_number += 1

    # ============================================================================
    # MONITORING AND RECOVERY PHASE
    # ============================================================================
    # This loop runs continuously until all jobs complete successfully.
    # Every 20 seconds it:
    #   1. Checks how many jobs are still running
    #   2. Scans log files for errors and cancels problematic jobs
    #   3. Checks each tracked job for completion or failure
    #   4. Resubmits jobs that failed or didn't complete (up to 3 times)
    # The loop ends only when no jobs are running (all completed or max retries reached)
    time.sleep(1)
    training_finished = False
    while not training_finished:
        num_running_jobs, _, _ = get_running_jobs(job_name, username)
        print(f"{num_running_jobs} jobs are running... Job: {job_name}")
        cancel_jobs_with_err_in_log(log_root, job_name, username)
        time.sleep(20)

        # resubmit unfinished jobs
        # Check each tracked job to see if it needs resubmission
        # A job needs resubmission if:
        # 1. It's not finished yet
        # 2. It has been resubmitted fewer than 3 times
        # 3. It's not currently in the SLURM queue (has stopped running)
        # 4. Either the result file is missing, incomplete, or indicates failure
        for k in list(meta_jobs.keys()):
            job_finished, job_file, result_file, resubmitted = meta_jobs[k]
            need_to_resubmit = False
            if not job_finished and resubmitted < 3:
                # check whether job is running
                if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
                    # check whether result file is finished?
                    if os.path.exists(result_file):
                        with open(result_file, "r", encoding="utf-8") as f_result:
                            evaluation_data = json.load(f_result)
                        progress = evaluation_data["_checkpoint"]["progress"]
                        # Check if progress is incomplete (e.g., progress = [0, 1] means 0 out of 1 completed)
                        if len(progress) < 2 or progress[0] < progress[1]:
                            need_to_resubmit = True
                        else:
                            # Even if progress says complete, check for failure conditions
                            for record in evaluation_data["_checkpoint"]["records"]:
                                if record["scores"]["score_route"] <= 0.00000000001:
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Agent couldn\'t be set up":
                                    need_to_resubmit = True
                                if record["status"] == "Failed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Simulation crashed":
                                    need_to_resubmit = True
                                if record["status"] == "Failed - Agent crashed":
                                    need_to_resubmit = True

                        if not need_to_resubmit:
                            # delete old job
                            print(f"Finished job {job_file}")
                            meta_jobs[k] = (True, None, None, 0)

                    else:
                        need_to_resubmit = True

            if need_to_resubmit:
                # Resubmission process:
                # 1. Preserve old log files by copying them to a timestamped directory
                # 2. Wait for a job slot to become available
                # 3. Submit the job again to SLURM
                # 4. Update meta_jobs with new job ID and incremented resubmit counter
                # 5. Mark old job ID as finished to clean up tracking
                routefile_number = Path(job_file).stem
                print(f"Resubmit job {routefile_number} (previous id: {k}). Waiting for jobs to finish...")

                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'max_num_jobs.txt'), 'r', encoding='utf-8') as f:
                    max_num_parallel_jobs = int(f.read())
                wait_for_jobs_to_finish(log_root, job_name, username, max_num_parallel_jobs)

                time_now_log = time.time()
                os.system(f'mkdir -p "{log_root}/run_files/logs_{routefile_number}_{time_now_log}"')
                os.system(f"cp {log_root}/run_files/logs/qsub_err{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}")
                os.system(f"cp {log_root}/run_files/logs/qsub_out{routefile_number}.log {log_root}/ \
                          run_files/logs_{routefile_number}_{time_now_log}")

                jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8").strip() \
                                                                                .rsplit(" ", maxsplit=1)[-1]
                meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                meta_jobs[k] = (True, None, None, 0)
                print(f"resubmitted job {routefile_number}. (new id: {jobid})")

        time.sleep(10)

        # Check if all jobs have finished
        # When no jobs are running, either all completed successfully or reached max resubmit attempts
        if num_running_jobs == 0:
            training_finished = True
