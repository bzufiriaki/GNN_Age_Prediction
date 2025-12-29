FROM tensorflow/tensorflow:2.10.1-gpu
RUN apt-get update
RUN apt-get install -y cmake-curses-gui swig gcc git g++ checkinstall python3-dev
WORKDIR /opt/project/


RUN apt-get install -y libsm6 libxext6 libxrender-dev git libgl1-mesa-glx
RUN pip3 install --use-feature=2020-resolver matplotlib pandas scikit-learn


ENV PYTHONPATH "${PYTHONPATH}:/opt/project/"