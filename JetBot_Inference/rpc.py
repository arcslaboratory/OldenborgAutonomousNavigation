import inspect
import json
import socket
from threading import Thread


SIZE = 1024  # Buffer size for receiving data from the socket

class RPCServer:
    """
    A Remote Procedure Call (RPC) server that handles client requests and executes registered functions.
    """
    def __init__(self, host: str = '0.0.0.0', port: int = 8033) -> None:
        """
        Initializes the RPC server with the provided host and port.
        """
        self.host = host  # Host IP address for the server
        self.port = port  # Port number for the server
        self.address = (host, port)
        self._methods = {}  # Dictionary to store registered methods that can be called by the client

    def registerMethod(self, function) -> None:
        """
        Registers a single function/method for remote invocation by clients.
        """
        try:
            self._methods.update({function.__name__: function})  # Store the function in the dictionary with its name as the key
        except:
            raise Exception('A non-function object has been passed into RPCServer.registerMethod(self, function)')

    def registerInstance(self, instance=None) -> None:
        """
        Registers all methods of an instance of a class for remote invocation by clients.
        """
        try:
            for functionName, function in inspect.getmembers(instance, predicate=inspect.ismethod):
                if not functionName.startswith('__'):
                    self._methods.update({functionName: function})
        except:
            raise Exception('A non-class object has been passed into RPCServer.registerInstance(self, instance)')

    def __handle__(self, client: socket.socket, address: tuple) -> None:
        """
        Handles incoming client requests by executing the requested function and sending back the result.
        """
        print(f'Managing requests from {address}.')
        while True:
            try:
                functionName, args, kwargs = json.loads(client.recv(SIZE).decode())  # Deserialize JSON data into function name, arguments, and keyword arguments
            except Exception as e:
                print(f'! Client {address} disconnected.')
                break
            # Showing request Type
            print(f'> {address} : {functionName}({args})')

            try:
                response = self._methods[functionName](*args, **kwargs)
            except Exception as e:
                # Send back exception if function called by client is not registered
                client.sendall(json.dumps(str(e)).encode())
            else:
                client.sendall(json.dumps(response).encode())

        print(f'Completed requests from {address}.')  # Log the completion of handling the client's requests
        client.close()  # Close the connection to the client

    def run(self) -> None:
        """
        Starts the RPC server and listens for incoming client connections.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # Create a TCP socket
            sock.bind(self.address)  # Bind the socket to the server address
            sock.listen()  # Start listening for incoming connections
            print(f'+ Server {self.address} running')  # Log the server start
            while True:
                try:
                    client, address = sock.accept()  # Accept a new client connection
                    Thread(target=self.__handle__, args=[client, address]).start()  # Handle the client connection in a new thread
                except KeyboardInterrupt:
                    print(f'- Server {self.address} interrupted')
                    break

class RPCClient:
    """
    A Remote Procedure Call (RPC) client that connects to an RPC server and invokes methods remotely.
    """
    def __init__(self, host: str = 'localhost', port: int = 8033) -> None:
        """
        Initializes the RPC client with the provided host and port.
        """
        self.__sock = None  # Socket to connect to the server
        self.__address = (host, port)  # Tuple containing the server's host and port

    def connect(self):
        """
        Connects the client to the RPC server.
        """
        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a TCP socket
            self.__sock.connect(self.__address)  # Connect to the server
        except EOFError as e:
            print(e)
            raise Exception('Client was not able to connect.')

    def disconnect(self):
        """
        Disconnects the client from the server.
        """
        try:
            self.__sock.close()  # Close the socket connection
        except:
            pass  # Ignore any exceptions that occur during disconnection

    def __getattr__(self, __name: str):
        """
        Dynamically handles method calls on the client by sending them to the server for execution.
        """
        def execute(*args, **kwargs):
            self.__sock.sendall(json.dumps((__name, args, kwargs)).encode()) # Send the function name, arguments, and keyword arguments as JSON
            response = json.loads(self.__sock.recv(SIZE).decode())   # Receive and deserialize the server's response
            return response

        return execute
