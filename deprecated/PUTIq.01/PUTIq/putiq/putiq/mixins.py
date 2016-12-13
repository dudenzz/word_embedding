

class LogMixin():

    LOG_LEVEL_LOG = 2
    LOG_LEVEL_VERBOSE = 3

    _log_level = LOG_LEVEL_LOG

    def _log(self, content, log_level=LOG_LEVEL_LOG):
        if self._log_level >= log_level:
            print(content)