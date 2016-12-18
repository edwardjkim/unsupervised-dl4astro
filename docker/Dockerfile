FROM jupyter/scipy-notebook

MAINTAINER Edward J Kim <edward.junhyung.kim@gmail.com>

USER root

RUN apt-get update && \
    apt-get -y -q install libfftw3-dev libatlas-base-dev libatlas-base-dev gfortran wget vim && \

    # SExtractor uses the LAPACK functions available in ATLAS, and
    # it won't be able to find LAPACK without the following line.
    update-alternatives --set liblapack.so /usr/lib/atlas-base/atlas/liblapack.so && \

    cd /tmp && \
    # install sextractor
    wget http://www.astromatic.net/download/sextractor/sextractor-2.19.5.tar.gz && \
    tar xvzf sextractor-2.19.5.tar.gz && \
    cd sextractor-2.19.5 && \
    ./configure --with-atlas-incdir=/usr/include/atlas && \
    make && \
    make install && \

    # Install Montage
    cd /tmp && \
    wget http://montage.ipac.caltech.edu/download/Montage_v4.0.tar.gz && \
    tar xvzf Montage_v4.0.tar.gz && \
    cd montage && \
    make

ENV PATH $PATH:/tmp/montage/bin

USER $NB_USER

RUN \
    # Python montage-wrapper
    pip3 install --upgrade pip && \
    pip3 install montage-wrapper && \

    pip3 install Theano Lasagne && \
    pip3 install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git
