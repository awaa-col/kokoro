__version__ = '0.9.4'

from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add custom handler with clean format including module and line number
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO" # "DEBUG" to enable logger.debug("message") and up prints 
                 # "ERROR" to enable only logger.error("message") prints
                 # etc
)

# Disable before release or as needed
logger.disable("kokoro")

# 【v4.6 终极循环导入修复】删除下面这两行。
# 它们试图在包的初始化阶段就去加载 model 和 pipeline，
# 而 model 又依赖包内的其他模块，导致了致命的循环导入。
# from .model import KModel
# from .pipeline import KPipeline
