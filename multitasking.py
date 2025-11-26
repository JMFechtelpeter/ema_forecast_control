import multiprocessing as mp
import subprocess
import re
import sys
import os
from typing import List, Tuple, Optional
import utils
import numpy as np
import torch as tc
import pandas as pd
import logging
from logging.handlers import QueueHandler, QueueListener


class Task:
    def __init__(self, command, name):
        self.command = command
        self.name = name


class Argument:
    def __init__(self, name: str, values: list, add_to_name_as: Optional[str]=None, 
                 infer_value_from: Optional[str]=None, infer_value_pattern: Optional[str]=None):
        self.name = name
        self.values = values
        self.add_to_name_as = add_to_name_as
        self.infer_value_from = infer_value_from
        self.infer_value_pattern = infer_value_pattern
        if len(values) > 1:
            print_statement = 'please specify a name addition for argument {}, because it has more than one value'.format(
                name)
            if name != 'run':
                assert add_to_name_as is not None, print_statement

class ArgumentTable:
    def __init__(self, name: str, for_argument: str, add_to_name_as: Optional[str]=None, table: Optional[pd.DataFrame]=None, file_path: Optional[str]=None):
        self.name = name
        self.for_argument = for_argument
        if file_path is not None:
            self.value_list = pd.read_csv(file_path, header=0)
        elif table is not None:
            self.value_list = table
        else:
            raise ValueError(f'ArgumentTable "{name}" needs either a table or a path to a csv file containing a table.')
        self.add_to_name_as = add_to_name_as

class ArgumentCombination:
    def __init__(self, name: tuple, values: list[tuple], add_to_name_as: tuple):
        self.name = name
        self.values = values
        self.add_to_name_as = add_to_name_as
        

def add_argument(tasks, arg: Argument|ArgumentTable|ArgumentCombination) -> List[Task]:
    new_tasks = []
    for task in tasks:
        if isinstance(arg, Argument):
            if arg.infer_value_from is not None and arg.infer_value_pattern is not None:
                # Infer value of this argument from other argument, e.g. Participant from data_path
                match = re.search(arg.infer_value_pattern, task.command)
                if match is not None:
                    arg_value = match.group(1)
                else:
                    raise ValueError(f'Argument value could not be inferred from {arg.infer_value_from}.')
                new_name = add_to_name(task, arg, arg_value)
                new_command = "  ".join([task.command, "--{}".format(arg.name), str(arg_value)])
                new_tasks.append(Task(new_command, new_name))
            else:
                # Declare this value in a normal way
                for arg_value in arg.values:
                    if type(arg_value) is list:
                        arg_name = '-'.join([str(i) for i in arg_value])
                        new_name = add_to_name(task, arg, arg_name)
                        arg_command = '  '.join([str(i) for i in arg_value])
                        new_command = "  ".join([task.command, "--{}".format(arg.name), arg_command])
                    else:
                        new_name = add_to_name(task, arg, arg_value)
                        new_command = "  ".join([task.command, "--{}".format(arg.name), str(arg_value)])
                    new_tasks.append(Task(new_command, new_name))
        elif isinstance(arg, ArgumentTable):
            match = re.search(rf'--{arg.for_argument}  (\w+)', task.command)
            if match is not None:
                for_value = match.group(1)
            else:
                raise ValueError(f'Argument {arg.for_argument}, to which this ArgumentList refers, has not been declared yet.')
            value_list = arg.value_list[for_value]
            value_list = value_list[value_list.notna()].to_list()
            for arg_value in value_list:
                if type(arg_value) is list:
                    arg_name = '-'.join([str(i) for i in arg_value])
                    new_name = add_to_name(task, arg, arg_name)
                    arg_command = '  '.join([str(i) for i in arg_value])
                    new_command = "  ".join([task.command, "--{}".format(arg.name), arg_command])
                else:
                    new_name = add_to_name(task, arg, arg_value)
                    new_command = "  ".join([task.command, "--{}".format(arg.name), str(arg_value)])
                new_tasks.append(Task(new_command, new_name))
        elif isinstance(arg, ArgumentCombination):
            for arg_value_tuple in arg.values:
                current_name = task.name
                current_command = task.command
                for k, arg_value in enumerate(arg_value_tuple):
                    if type(arg_value) is list:
                        arg_name = '-'.join([str(i) for i in arg_value])
                        if len(current_name)==0:
                            current_name = "_".join([arg.add_to_name_as[k], str(arg_value).zfill(2)])
                        else:
                            current_name = "_".join([current_name, arg.add_to_name_as[k], str(arg_value).zfill(2)])
                        arg_command = '  '.join([str(i) for i in arg_value])
                        current_command = "  ".join([current_command, "--{}".format(arg.name[k]), arg_command])
                    else:
                        if len(current_name)==0:
                            current_name = "_".join([arg.add_to_name_as[k], str(arg_value).zfill(2)])
                        else:
                            current_name = "_".join([current_name, arg.add_to_name_as[k], str(arg_value).zfill(2)])
                        current_command = "  ".join([current_command, "--{}".format(arg.name[k]), str(arg_value)])
                new_tasks.append(Task(current_command, current_name))
            
    return new_tasks


def add_to_name(task, arg, arg_value):
    if arg.add_to_name_as is not None:
        if arg.name == 'data_path':
            arg_value = os.path.split(arg_value)[1].zfill(2)
        else:
            arg_value = str(arg_value).zfill(2)
        if len(task.name)==0:
            new_name = "_".join([arg.add_to_name_as, arg_value])
        else:
            new_name = "_".join([task.name, arg.add_to_name_as, arg_value])
    else:
        new_name = task.name
    return new_name


def check_arguments_for_gpu(args: List[Argument]) -> bool:
    '''
    Check the ubermain arguments for the 'use_gpu' flag
    and general cuda availability.
    '''
    use_gpu = False
    # if the user specifies device ids himself,
    # do not bother distributing the tasks.
    for arg in args:
        if arg.name == 'device_id':
            print("Device id(s) specified by user "
                  "-> manual task distribution")
            return False
        elif arg.name == 'use_gpu':
            if 1 in arg.values:
                assert tc.cuda.is_available(),  \
                    "CUDA is not available."
                print("'use_gpu' flag is set.")
                use_gpu = True
    if use_gpu and tc.cuda.is_available():
        print("Will distribute tasks to GPUs "
              "automatically.")
    return use_gpu


def distribute_tasks_across_gpus(tasks: List[Task],
                                 n_proc_per_gpu: int,
                                 n_cpu: int) -> Tuple[List, int]:
    '''
    Checks current GPU utilization of the machine,
    picks out idle devices and distributes them
    across tasks.
    '''
    util_dict = utils.get_current_gpu_utilization()
    # filter device ids of unused GPUs
    device_ids = []
    for id_, util in util_dict.items():
        if util < 75:
            device_ids.append(id_)

    if not device_ids:
        raise RuntimeError("All GPUs of the machine are in use!")

    # check if there are too many parallel processes spawned by user
    # compared to available GPUs
    device_distribution = np.repeat(device_ids, min(n_cpu, n_proc_per_gpu))
    sz = device_distribution.size
    if sz < n_cpu:
        print("There are not enough GPU Resources available to spawn "
              f"{n_cpu} processes. Reducing number of parallel runs "
              f"to {sz}")
        new_n_cpu = sz
    else:
        new_n_cpu = n_cpu

    # distribute devices across tasks
    new_tasks = []
    idx = 0
    for task in tasks:
        arg = Argument('device_id', [device_distribution[idx]])
        new_tasks.append(*add_argument([task], arg))
        idx += 1
        if idx == new_n_cpu:
            idx = 0
    return new_tasks, new_n_cpu


def create_tasks_from_arguments(args: List[Argument], n_proc_per_gpu: int, n_cpu: int,
                                python_path: Optional[str]=None, main_path: Optional[str]=None) -> Tuple[List, int]:
    # check args for gpu usage
    use_gpu = check_arguments_for_gpu(args)
    if python_path is None:
        python_path = sys.executable
    if main_path is None:
        main_path = 'main.py'
    tasks = [Task(command=f'{python_path}  {main_path}', name='')]
    for arg in args:
        tasks = add_argument(tasks, arg)

    pp = n_cpu
    if use_gpu:
        # tasks w/ device id, number of parallel processes
        tasks, pp = distribute_tasks_across_gpus(tasks,
                                                 n_proc_per_gpu,
                                                 n_cpu)

    for k, task in enumerate(tasks):
        pbar_descr = f'Job {k}/{len(tasks)}'
        task.command = "  ".join([task.command, '--name', task.name, '--pbar_descr', pbar_descr])

    return tasks, pp

#logging inspired by https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python

def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)

def logger_init():
    q = mp.Queue()
    # this is the handler for all log records
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)

    return ql, q


def run_settings(tasks, n_cpu):
    # q_listener, q = logger_init()
    # logging.info('Running %i jobs, %i in parallel.', len(tasks), n_cpu)
    # pool = mp.Pool(processes=n_cpu, initializer=worker_init, initargs=[q])
    pool = mp.Pool(processes=n_cpu)
    pool.map(process_task, tasks, chunksize=1)
    pool.close()
    pool.join()
    # q_listener.stop()

def process_task(task):
    subprocess.call(task.command.split('  '))
