from logging.config import dictConfig

from vllm.logger import DEFAULT_LOGGING_CONFIG


def register():
    """Register the Spyre platform."""
    return "vllm_spyre.platform.SpyrePlatform"


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config = {**DEFAULT_LOGGING_CONFIG}

    # Copy the vLLM logging configurations for our package
    config["formatters"]["vllm_spyre"] = DEFAULT_LOGGING_CONFIG["formatters"][
        "vllm"]

    handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
    handler_config["formatter"] = "vllm_spyre"
    config["handlers"]["vllm_spyre"] = handler_config

    logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
    logger_config["handlers"] = ["vllm_spyre"]
    config["loggers"]["vllm_spyre"] = logger_config

    dictConfig(config)


_init_logging()
