FROM gcc:latest

# Install required tools (CMake, GDB, Make, etc.)
RUN apt-get update && apt-get install -y cmake gdb make ninja-build valgrind
COPY exec-valgrind.sh /usr/local/bin/exec-valgrind.sh

# Create a workspace directory
RUN mkdir -p /workspace

# Set the workspace directory as the default
WORKDIR /workspace
