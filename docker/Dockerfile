FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
ENV DEBIAN_FRONTEND "noninteractive"

RUN apt-get update
RUN apt-get -y \
	-o Dpkg::Options::="--force-confdef" \
	-o Dpkg::Options::="--force-confold" dist-upgrade
RUN apt-get install -y zsh silversearcher-ag tmux git curl wget build-essential python-dev libgtk2.0-dev vim
RUN useradd -m -d /home/ubuntu ubuntu
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ubuntu:ubuntu /home/ubuntu
RUN chsh -s /usr/bin/zsh ubuntu

USER ubuntu
WORKDIR /home/ubuntu
ENV HOME /home/ubuntu

RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/ubuntu/.zshrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.zshrc
RUN echo 'export CPATH=$HOME/cuda/include:$CPATH' >> /home/ubuntu/.zshrc
RUN echo 'export LIBRARY_PATH=$HOME/cuda/lib64:$LIBRARY_PATH' >> /home/ubuntu/.zshrc
RUN echo 'export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.zshrc

RUN git clone https://github.com/yyuu/pyenv.git /home/ubuntu/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/ubuntu/.zshrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/ubuntu/.zshrc
RUN echo 'eval "$(pyenv init -)"' >> /home/ubuntu/.zshrc
RUN ls -la ~/
RUN chown -R ubuntu:ubuntu /home/ubuntu

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)"
RUN pyenv install anaconda3-4.2.0
RUN pyenv global anaconda3-4.2.0
RUN pyenv rehash
RUN conda install -c https://conda.binstar.org/menpo -y opencv3

RUN git clone https://github.com/pfnet/chainer
RUN cd chainer; python setup.py install
