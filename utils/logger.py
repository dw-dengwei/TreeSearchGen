import logging
import colorlog


log_colors_config = {
  'DEBUG': 'cyan',
  'INFO': 'white',
  'WARNING': 'yellow',
  'ERROR': 'red',
  'CRITICAL': 'bold_red',
}

# Create logger
logger = logging.getLogger(__name__)

# Create handler
handler = logging.StreamHandler()

# Create formatter
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)-23s - %(name)-15s [%(filename)20s:%(lineno)-4d] - %(levelname)s - %(message)s',
    log_colors=log_colors_config
)

# Add formatter to handler
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

# Set log level
logger.setLevel(logging.INFO)

# Prevent log propagation to avoid duplicate messages
logger.propagate = False

# Export logger instance
__all__ = ['logger'] 