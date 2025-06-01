"""
Streamline launching of tensorboard on an available port.
"""

import argparse
import os
import random

def is_port_in_use(port):
    """
    https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
def main(args):
    port_options = [6000 + i for i in range(50)]
    port_options += [60000 for i in range(50, 100)]
    # randomize the port options
    random.shuffle(port_options)
    for port in port_options:
        if not is_port_in_use(port):
            print(f"Using port: {port}")
            break
    print("Starting TensorBoard...")
    os.system(f"tensorboard --logdir {args.log_dir} --port {port} --bind_all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorBoard Logger for EchoDARENet")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Directory to save TensorBoard logs")
    args = parser.parse_args()
    main(args)

    