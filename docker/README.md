# Create Docker image

## Install docker

```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
sudo sh -c "echo 'deb https://apt.dockerproject.org/repo ubuntu-trusty main' >> /etc/apt/sources.list.d/docker.list"
sudo apt-get update
sudo apt-get purge lxc-docker
sudo apt-get install -y linux-image-extra-$(uname -r) linux-image-extra-virtual
sudo apt-get install -y docker-engine
sudo service docker start
sudo groupadd docker
sudo gpasswd -a $USER docker
```

## Install NVIDIA-Docker

```
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```

## Build Docker image

```
sudo nvidia-docker build -t chainer-segnet .
```

## Launch the docker container

Move to the top of this project dirtree, then

```
sudo nvidia-docker run \
-v $PWD:/home/ubuntu/chainer-segnet \
-p 8888:8888 \
-ti chainer-segnet /usr/bin/zsh
```
