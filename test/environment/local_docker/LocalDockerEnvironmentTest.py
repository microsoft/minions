"""Simple manual test script for LocalDockerEnvironment.

This file demonstrates:
  1. Creating multiple containerized shell environments with different mount permissions.
  2. Executing commands through the HTTP bridge exposed by the FastAPI shell server.
  3. Expected behavior differences between READ_WRITE and READ_ONLY mounts.

NOTES:
  - Each environment must use a distinct host port to avoid binding conflicts.
  - The container internally listens on port 8080; host ports map to that internal port.
  - Commands like `cd` affect only the shell session state inside the container serving that env.
  - A READ_ONLY mount should prevent file creation (e.g. "touch should_fail.txt") inside the mounted path.
  - This script does not automatically stop containers so you can inspect them afterward.
    Remember to call `env.stop()` (or prune with Docker) when done to free resources.
  - Error handling here is minimal; in production wrap execute calls and inspect return values.

USAGE:
    python -m test.environment.local_docker.LocalDockerEnvironmentTest

Clean Up Manually (example):
    docker ps | grep <image_name>
    # then stop/remove as needed
"""

from microbots.environment.local_docker import LocalDockerEnvironment


def LocalDockerEnvironmentTest():
    # Environment 1: Read-write mount of /home/kkaitepalli/MAP on host port 8085
    # Provide absolute path to a directory on your host machine
    env1 = LocalDockerEnvironment(
        port=8085,
        folder_to_mount="/home/kkaitepalli/MAP",
        permission="READ_WRITE",
    )

    # Environment 2: No mount (isolated filesystem view) on host port 8086
    env2 = LocalDockerEnvironment(port=8086)

    # Environment 3: Read-only mount of /home/kkaitepalli/telescope on host port 8087
    env3 = LocalDockerEnvironment(
        port=8087,
        folder_to_mount="/home/kkaitepalli/telescope",
        permission="READ_ONLY",
    )

    try:
        # Navigate inside env1 into the mounted directory
        response = env1.execute("cd MAP")
        print("env1 cd MAP:", response)

        # List contents to verify mount
        response = env1.execute("ls -la")
        print("env1 ls -la:\n", response)

        # Create a file in env2's working directory (no mount involved)
        env2.execute("touch testfile.txt")

        # Change to subdirectory in env1 (assuming it exists in the mounted content)
        env1.execute("cd mariner-aks-pipelines")

        # Attempt navigation inside env3's read-only mount
        env3.execute("cd telescope")

        # Attempt to create a file in read-only environment (should fail)
        response = env3.execute("touch should_fail.txt")
        print("env3 touch should_fail.txt (expected failure or error msg):", response)

        # Show current working directory in env1
        response = env1.execute("pwd")
        print("env1 pwd:", response)

    finally:
        print(
            "Containers left running for inspection. Call envX.stop() or \n"
            "docker stop <id> && docker rm <id> when finished."
        )
        # Example cleanup (uncomment as needed):
        # env1.stop(); env2.stop(); env3.stop()


if __name__ == "__main__":  # Allows running via: python -m test.environment.local_docker.LocalDockerEnvironmentTest
    LocalDockerEnvironmentTest()
