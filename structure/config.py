from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

def load_config():

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid configuration file.")

    return config


CONFIG = load_config()


def get_path(key):

    path = Path(CONFIG["paths"][key])

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


INPUT_FILE = get_path("input_catalog")

OUTPUT_DIR = get_path("output_directory")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)