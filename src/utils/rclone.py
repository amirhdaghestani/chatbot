"""This module implements rclone."""
import time
import subprocess
import tempfile
from typing import List

from logger.ve_logger import VeLogger

# This repository is forked from https://github.com/ddragosd/python-rclone
RCLONE_PROGRESS_INTERVAL = 0.1


# TODO: add logger to class, timeout if server not accessable ((https://jira.mci.dev/browse/VE-302))
class RClone():
    """
    Wrapper class for rclone.
    """

    def __init__(self, cfg):
        """

        Args:
            cfg ([str]): rclone config

        >>>
        >>>
        >>>
        """
        self.cfg = cfg.replace("\\n", "\n")
        self.log = VeLogger()

    def _execute(self, command_with_args):
        """
        Execute the given `command_with_args` using Popen
        Args:
            - command_with_args (list) : An array with the command to execute,
                                         and its arguments. Each argument is given
                                         as a new element in the list.
        """
        self.log.debug("Invoking : %s", " ".join(command_with_args))
        try:
            with subprocess.Popen(
                    command_with_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE) as proc:

                self.log.info("start rclone")
                err_lines = ''
                out_lines = ''
                for line in iter(proc.stdout.readline, b''):
                    line = line.decode("utf-8")
                    out_lines += line
                    time.sleep(RCLONE_PROGRESS_INTERVAL)

                for line in iter(proc.stderr.readline, b''):
                    line = line.decode("utf-8")
                    self.log.error(line)
                    err_lines += line
                    time.sleep(RCLONE_PROGRESS_INTERVAL)

                return {
                    "code": proc.returncode,
                    "out": out_lines,
                    "error": err_lines
                }
        except FileNotFoundError as not_found_e:
            self.log.error("Executable not found. %s", not_found_e)
            return {
                "code": -20,
                "error": not_found_e
            }
        except Exception as generic_e:
            self.log.error("Error running command. Reason: %s", generic_e)
            return {
                "code": -30,
                "error": generic_e
            }

    def run_cmd(self, command: List[str], extra_args=[]):
        """
        Execute rclone command
        Args:
            - command (string): the rclone command to execute.
            - extra_args (list): extra arguments to be passed to the rclone command
        """
        # save the configuration in a temporary file
        with tempfile.NamedTemporaryFile(mode='wt', delete=True) as cfg_file:
            # cfg_file is automatically cleaned up by python
            cfg_file.write(self.cfg)
            cfg_file.flush()

            command_with_args = ["rclone", *command, "--config", cfg_file.name]
            command_with_args += extra_args
            command_result = self._execute(command_with_args)
            cfg_file.close()
            return command_result

    def copy(self, source, dest, flags=[]):
        """
        Executes: rclone copy source:path dest:path [flags]
        Args:
        - source (string): A string "source:path"
        - dest (string): A string "dest:path"
        - flags (list): Extra flags as per `rclone copy --help` flags.
        """
        return self.run_cmd(command=["copy", "-P"], extra_args=[source] + [dest] + flags)

    def sync(self, source, dest, flags=[]):
        """
        Executes: rclone sync source:path dest:path [flags]
        Args:
        - source (string): A string "source:path"
        - dest (string): A string "dest:path"
        - flags (list): Extra flags as per `rclone sync --help` flags.
        """
        return self.run_cmd(command=["sync"], extra_args=[source] + [dest] + flags)

    def listremotes(self, flags=[]):
        """
        Executes: rclone listremotes [flags]
        Args:
        - flags (list): Extra flags as per `rclone listremotes --help` flags.

        """
        return self.run_cmd(command=["listremotes"], extra_args=flags)

    def ls(self, dest, flags=[]):
        """
        Executes: rclone ls remote:path [flags]
        Args:
        - dest (string): A string "remote:path" representing the location to list.
        """
        return self.run_cmd(command=["ls"], extra_args=[dest] + flags)

    def lsjson(self, dest, flags=[]):
        """
        Executes: rclone lsjson remote:path [flags]
        Args:
        - dest (string): A string "remote:path" representing the location to list.
        """
        return self.run_cmd(command=["lsjson"], extra_args=[dest] + flags)

    def delete(self, dest, flags=[]):
        """
        Executes: rclone delete remote:path
        Args:
        - dest (string): A string "remote:path" representing the location to delete.
        """
        return self.run_cmd(command=["delete"], extra_args=[dest] + flags)
