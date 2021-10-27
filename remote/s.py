# imagine an ability wherein any package can be used by client, cloud or local

import os
import inspect
import requests
import traceback

from types import SimpleNamespace

from starlette.requests import Request
from starlette.responses import Response

import uvicorn
from pydantic import create_model
from fastapi import FastAPI

ALL_ENDPOINTS = []
URL = os.getenv("GPERC_URL", "http://localhost:6969")
USAGE_MODE = os.getenv("GPERC_MODE", "local") # is this local or client
# assert USAGE_MODE in ["local", "client"], f"{USAGE_MODE} must be one of 'local' or 'client'"

# wrapper that converts any function to a cloud function by assinging it as url endpoint
def remote(path, request, status_code=None, name=None, message = (str, None), **_response_fields):
  path = path.rstrip("/")  # remove trailing slash
  assert path[0] == "/", "path must start with /"
  method_name = name

  # add "message" and enforce tuple-ing of response_fields values
  _response_fields.update({"message": message})
  response_fields = {k: tuple(v) for k, v in _response_fields.items()}

  def wrapper(fn):
    # spec gives you the full annotation of the function, what goes in what comes out, etc.
    # here we take this spec and extract or remove things from it. If I have a function like this:
    #
    # >>> def fn() -> str:
    #
    # So we do two things here that make life a little bit easier:
    # 1. we enforce typing on all the kwargs, if the user does it wrong, it's their problem
    #    not really ours.
    spec = inspect.getfullargspec(fn)
    spec.annotations.pop("return", None)
    assert len(spec.annotations) == len(
      spec.args
    ), f"All keys must be annotated, these are not: {set(spec.args).difference(set(spec.annotations))}"

    # after a bit of looking at the source code for FastAPI, first of all, amazing codebase,
    # meant to be read not documented. Secondly, it takes in a field_definitions which should
    # have the structure: { keyword: [type, value], ... } so we create such a dict
    field_definitions = {k: [t, None] for k, t in spec.annotations.items()}
    required_fields = set([k for k, _ in field_definitions.items()])
    defaults = {}
    if spec.defaults:
      for k, v in zip(spec.args[::-1], spec.defaults[::-1]):
        field_definitions[k][1] = v
        defaults[k] = v
        required_fields.remove(k)
    field_definitions = {k: tuple(v) for k, v in field_definitions.items()}

    name = path.replace("/", "_")[1:].capitalize()
    request_model = create_model(f"{name}_Request", **field_definitions)
    response_model = create_model(f"{name}_Response", **response_fields)

    # create a function that takes in a request and returns a response
    def fn_wrapper(request: Request, response: Response, input_data: request_model = None):
      # # check that all required fields are present in input_data
      # diff = required_fields.difference(input_data.dict().keys())
      # print(diff)
      # print(required_fields)
      # print(input_data.dict().keys())
      # if len(diff) > 0:
      #   response.status_code = 400
      #   return response_model(message = f"Fields not found: {diff}")

      # give it to the model
      input_data = {} if input_data == None else input_data.dict()
      try:
        out = fn(**input_data)
      except Exception as e:
        _t = traceback.format_exc()
        response.status_code = 500
        return response_model(message=f"Command '{fn.__name__}' not executed | {e} | {_t}")

      # convert the output to a response
      # print("--->>>", out)
      if isinstance(out, dict):
        out_dict = out
      elif out == None:
        return response_model(message = "generic-success")
      else:
        if len(_response_fields) == 1:  #
          out_dict = {list(_response_fields.keys())[0]: out}
        else:
          out_dict = {k: o for k, o in zip(_response_fields.keys(), out)}
      response.status_code = status_code if status_code else 200
      return response_model(**out_dict)

    req = request.upper()
    assert req in [
      "GET",
      "POST",
      "PUT",
      "DELETE",
      "PATCH",
    ], f"{req} is not a valid request, must be one of GET, POST, PUT, DELETE, PATCH"

    # check if path already exists
    for endpoint in ALL_ENDPOINTS:
      if endpoint["path"] == path:
        raise Exception(f"path {path} already exists")

    # add an endpoint dict that can be directly loaded into app.add_api_route()
    ALL_ENDPOINTS.append({
      "path": path,
      "endpoint": fn_wrapper,
      "status_code": status_code,
      "response_model": response_model,
      "name": method_name,
      "methods": [req],
    })

    def __func(*args, **kwargs):
      if USAGE_MODE == "local":
        # this function lives on the local machine and thus has to be used
        # like a normal function, this is the default mode
        return fn(*args, **kwargs)

      elif USAGE_MODE == "client":
        # take the inputs and convert it into a pydantic.BaseModel like thing
        # request the cloud function with the converted data and get the output
        # convert the output into python objects and return it.
        # I have to populate request_model with the incoming args and kwargs
        raw_dict = defaults.copy()
        raw_dict.update({k: v for k, v in zip(spec.args, args)})
        raw_dict.update(kwargs)

        # assert that raw_dict is a subset of field_definitions
        for k, v in raw_dict.items():
          if k not in field_definitions:
            raise Exception(f"{k} is not a valid field")

        # hit the self-api and structure the output
        try:
          r = getattr(requests, request.lower())(f"{URL}" + path, json=raw_dict)
          # out = SimpleNamespace(**r.json())
          return r.json()
        except Exception as e:
          raise Exception(f"{e}")

        return Non
    return __func
  return wrapper



def become_server(
  host: str = "0.0.0.0",
  port: int = 6969,
  reload: bool = False,
  log_level: str = "warning"
):
  """
  all the instructions here are for CLI, convert to non-CLI: https://www.uvicorn.org/settings/

  Args:
      host (str): host location
      port (int): port to expose to
      reload (bool, optional): if true, sets up a watcher good while developing
      log_level (str, optional): one of "critical","error","warning","info","debug","trace",
  """
  os.environ['GPERC_MODE'] = 'server' # force this to server mode because other services might use this
  global USAGE_MODE
  USAGE_MODE = "server"

  # create a ping point that makes it for testing
  def __ping_method(r: Request, req: create_model(f'Ping_Request', message = (str, None)) = None):
    if req == None:
      return {'message': 'pong'}
    return {"message": "pong" if req.message == None else req.message}
  response_model = create_model(f'Ping_Response', message = (str, None))

  ALL_ENDPOINTS.append({
    "path": "/",
    "endpoint": __ping_method,
    "status_code": 200,
    "response_model": response_model,
    "name": None,
    "methods": ["GET"]
  })

  app = FastAPI()
  for item in ALL_ENDPOINTS:
    app.add_api_route(**item)
  uvicorn.run(app, host = host, port = port, reload = reload, log_level = log_level)

# keeping a seperate function out here just to test if everything works or not
@remote("/webserver/status", "GET", status = (list, None))
def status():
  all_methods = []
  for a in ALL_ENDPOINTS:
    all_methods.append({
      "path": a["path"],
      "methods": a["methods"],
      "endpoint": a["endpoint"].__name__,
      "status_code": a["status_code"]
    })
  return {"status": all_methods}
