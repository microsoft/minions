import socket


def get_free_port():
    # Create a temporary socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Bind the socket to address 0.0.0.0 and port 0
        # Port 0 tells the OS to assign a free ephemeral port
        sock.bind(("0.0.0.0", 0))
        # Get the port number that was assigned
        port = sock.getsockname()[1]
        return port
    finally:
        # Close the socket to release the port
        sock.close()
