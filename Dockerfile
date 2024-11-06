# Use PyTorch base image
FROM pytorch/pytorch:latest

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    sudo \
    make \
    g++ \
    zsh \
    && chsh -s /bin/zsh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*   # cleanup (smaller image)

# Configure a non-root user with sudo privileges
ARG USERNAME=developer  # Change this to preferred username
ARG USER_UID=1001
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

# Set working directory
WORKDIR /home/$USERNAME/unit-scaling

# Puts pip install libs on $PATH & sets correct locale
ENV PATH="$PATH:/home/$USERNAME/.local/bin" \
    LC_ALL=C.UTF-8

# Install Python dependencies
COPY requirements-dev.txt .
RUN pip install --user -r requirements-dev.txt

# Creates basic .zshrc
RUN sudo cp /etc/zsh/newuser.zshrc.recommended /home/$USERNAME/.zshrc

CMD ["/bin/zsh"]
