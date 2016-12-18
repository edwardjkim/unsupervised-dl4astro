# Dockerfile

Updated from [dl4astro](https://github.com/EdwardJKim/dl4astro/blob/master/docker/Dockerfile).

Build with
```shell
$ docker build -t <image name>
```

and run with

```shell
$ docker run -d --name <container name> -p 8888:8888 -e "PASSWORD=YourPassword" -v /mnt/volume:/home/jovyan/work/shared <image name>
```

Access the notebook server at `http://<your ip address>:8888`.
