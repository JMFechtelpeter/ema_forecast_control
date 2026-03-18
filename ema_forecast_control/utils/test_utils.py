import sys
import platform

def print_machine_name():
    machine_name = platform.node()
    print(f"Machine name: {machine_name}", file=sys.stdout)