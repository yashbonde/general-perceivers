from fire import Fire
from .cli import Main, Serve

if __name__ == "__main__":
    Fire({"main": Main, "serve": Serve})
