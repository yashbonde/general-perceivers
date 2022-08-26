from dataclasses import dataclass
import io
import os
import re
import subprocess
from glob import glob
from time import sleep
from tempfile import gettempdir

from gperc.data.trainer import Trainer
from gperc.models import Perceiver
from gperc.configs import PerceiverConfig
from gperc.data.arrow import ArrowConsumer

from nbox import Instance
from nbox.utils import get_random_name

LS_REGEX = re.compile(r"([dwrx-]+)\s+.*\d{2,}\s(\w{3}\s\d+\s\d+:\d+)\s(.*)")

class __T(Trainer):
  def __init__(self, model_config: PerceiverConfig, save_folder = None, save_every = 1000, client = None):
    model = Perceiver(model_config)
    super().__init__(model, save_folder, save_every, client)
    self.model_config = model_config


class Tera:
  def __init__(
    self,
    model_configs,
    data_configs,
    trainer_configs,
  ):
    # check if the type of all the configs is correct
    if not isinstance(model_configs, list):
      model_configs = [model_configs]
    if not isinstance(data_configs, list):
      data_configs = [data_configs]
    if not isinstance(trainer_configs, list):
      trainer_configs = [trainer_configs]

    assert len(model_configs) == len(data_configs) == len(trainer_configs), \
      "model [{}], data [{}] and trainer [{}] configs must have the same length".format(
        len(model_configs), len(data_configs), len(trainer_configs)
      )

    self.model_configs = model_configs
    self.data_configs = data_configs
    self.trainer_configs = trainer_configs
    self.triplets = list(zip(model_configs, data_configs, trainer_configs))

  def run(self, instance: Instance, jobs = 1, poll_time = 5):
    """Run the trainer on multiple trainer jobs.
      
      Args:
        instance (nbox.Instance): The machine to clone
        jobs (int): The number of jobs to run in parallel
    """
    root = instance.name
    project_instances = [instance]
    for i in range(jobs - 1):
      suffix = get_random_name(True).split("-")[0]
      project_instances.append(instance.clone(f"{root}-{i}-{suffix}"))

    # instance starting is blocking, start in threads
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers = jobs) as executor:
      def _start(instance, **start_kwargs):
        instance.start(**start_kwargs)
        return instance

      future_to_instance = {
        executor.submit(
          _start,
          instance = instance,
          **start_kwargs
        ): instance
        for start_kwargs, instance in zip(self.start_kwargs, project_instances)
      }
      for future in as_completed(future_to_instance):
        instance = future_to_instance[future]
        try:
          future.result()
        except Exception as e:
          print(f"{instance.name} failed to start: {e}")

    config_paths = []
    for model_config, data_config, trainer_config in self.triplets:
      trainer = __T(
        model_config = model_config,
        data_config = data_config,
        trainer_config = trainer_config,
        project_instances = project_instances
      )
      job_name = get_random_name()
      fpath = os.path.join(gettempdir(), job_name)
      trainer.serialise(fpath)
      config_paths.append(fpath)

    done = False
    instance_to_pid_done = {
      instance: [None, None] for instance in project_instances
    }
    configs_left = config_paths.copy()
    while not done:
      for instance in project_instances:
        pid, name = instance_to_pid_done[instance]
        status, err = instance(pid)
        if status != "running" or pid == None:
          if err != None:
            # this run errored out
            instance.mv(f"nbx://{name}/stdout_err.log", f"./{name}-stdout_err.log")
            print(f"{instance.name} failed to run {name}: {err}. Check: {f'./{name}-stdout_err.log'}")

          if len(config_paths):
            # run a new job
            name = get_random_name()
            instance.mv(configs_left.pop(0), f"nbx://{name}")
            pid = instance.run(f"./train.py --config-path nbx://{name}")
            instance_to_pid_done[instance] = [pid, name]
          else:
            instance.stop()
            instance_to_pid_done[instance] = [None, None]

      done = True
      for instance, pid in instance_to_pid_done.items():
        if pid == None:
          print(f"{instance.name} is not running any code, shutting down")
          instance.stop()
        if pid != None:
          done = False
          break

      sleep(poll_time)
