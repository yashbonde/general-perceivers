import os
import hashlib
from fire import Fire

import concurrent.futures

from s import become_server, remote
from model import GrokkingConfig, GrokkingPerceiver, GrokkingTrainer

import threading


def thread_it(run_function):
    def wrapper(*args, **kwargs):
        temp = threading.Thread(target=run_function, args=args, kwargs=kwargs)
        temp.setDaemon(True)
        temp.start()
        return temp

    return wrapper


class Orchestrator:
    r"""Orchestrator is a fancy CRUD class with method triggers that direct the individual
    instance. All the functions follow a Golang style syntax which returns the function output
    and err for every method. This is a great because it forces the output to be returned to
    the user incase something fails."""

    def __init__(self, max_proc=10):
        print(f"> PID: {os.getpid()}")
        self.instances = []
        self.i_hashes = []
        self.max_proc = max_proc

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_proc,
            thread_name_prefix="gperc-orchestrator-thread-",
        )
        self.futures = {}
        self.results = {}

    # C: Create
    def create_instance(self, instance):
        r"""this method is simply used to create an instance, the instance can be anything that has
        to be managed. This can be a model, a dataset donwloader, etc
        """
        if len(self.instances) > self.max_proc:
            print(f"> Queue already full: {len(self.instances)}")
            return None, "queue full"

        # hash(instance) is not a stable approach, so we take hash of dict
        # https://death.andgravity.com/stable-hashing
        config = instance.config
        rv = {}
        for k, v in config.__dict__.items():
            if not isinstance(v, str):
                v = str(v)
            rv[k] = v
        rv = {k: v for k, v in sorted(rv.items())}  # sort to ensure it's always the same
        i_hash = hashlib.md5(str(rv).encode("utf-8")).hexdigest()

        if i_hash not in self.i_hashes:
            self.instances.append(instance)
            self.i_hashes.append(i_hash)
            print(f"> Instance added: {i_hash}")
        else:
            print(f"> Instance already exists: {i_hash}")
        return i_hash, None

    # R: Read
    def read(self, i_hash):
        pass

    # U: Update
    def update(self, i_hash, fn, *args, **kwargs):
        print(f"> Updating instance: {i_hash} and function: {fn.__name__}")
        # future = self.executor.submit(fn, *args, **kwargs)
        future = thread_it(fn)(*args, **kwargs)
        self.futures[i_hash] = future
        return None, "instance updated"

    # once the instance has been updated, we can get the status using Futures
    def get_status(self, i_hash):
        """What is the status of the current instance, obtained from"""
        if i_hash not in self.futures:
            return None, f"There is nothing running against id: {i_hash}"

        f = self.futures[i_hash]
        if f.running():
            return None, f"Instance id: {i_hash} is running"

        if f.done():
            self.results[i_hash] = f.result()
            return None, f"Instance id: {i_hash} is done"

    # D: Delete
    def delete_instance(self, i_hash):
        if i_hash in self.i_hashes:
            self.instances.pop(self.i_hashes.index(i_hash))
            self.i_hashes.remove(i_hash)
            print(f"> Instance removed: {i_hash}")
            return None, "instance removed"
        else:
            print(f"> Instance doesn't exist: {i_hash}")
            return None, "instance does not exist"


O = Orchestrator()

# ------ define functions that will be used everywhere ------ #
@remote("/orchestrator/status", "GET", n_models=(int, None), max_proc=(int, None), instances=(list, None))
def status():
    return {"n_models": len(O.instances), "max_proc": O.max_proc, "instances": O.i_hashes}


@remote("/create_model", "POST", model_id=("str", None))
def create_model(latent_dim: int, modulo: int, max_len: int, latent_frac: float = 1):
    config = GrokkingConfig(latent_dim, modulo, max_len, latent_frac)
    model = GrokkingPerceiver(config)
    i_hash, err = O.create_instance(model)
    if err != None:
        return {"message": err}
    else:
        return {"model_id": i_hash}


@remote("/delete_model", "POST")
def delete_model(model_id: str):
    _, message = O.delete_instance(model_id)
    return {"message": message}


@remote("/train_model", "POST")
def train_model(
    model_id: str,
    func_name: str,
    split_perc: float = 0.9,
    num_steps: int = int(1e4),
    batch_size: int = 528,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
):
    model = O.instances[O.i_hashes.index(model_id)]
    trainer = GrokkingTrainer(
        model_id,
        model,
        split_perc=split_perc,
        func=func_name,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_steps=num_steps,
    )
    out, message = O.update(model_id, trainer.train)
    return {"message": message}


if __name__ == "__main__":
    Fire({"serve": become_server})
